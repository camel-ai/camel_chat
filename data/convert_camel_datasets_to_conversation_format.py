import glob
import json
import os
import random
import uuid

from tqdm.auto import tqdm


def convert_filelist_to_dataset(file_list):
    conv_dataset = []
    for file in tqdm(file_list):
        with open(file) as f:
            contents = json.load(f)
        contents = {k.lower(): v for k, v in contents.items()}
        message_keys = [key for key in contents.keys() if 'message_' in key]
        filtered_content = filter_keywords(contents, message_keys)
        conv_dict = convert_to_chat_format(filtered_content, message_keys)
        conv_dataset = conv_dataset + [conv_dict]
    return conv_dataset

def filter_keywords(contents, message_keys):
    if contents["termination_reason"] == "<CAMEL_TASK_DONE>":
        contents.pop(message_keys[-1])
        message_keys.pop()
        contents["num_messages"] = contents["num_messages"] - 1
    
    keywords = ["Instruction: ", "\nInput: None", "Input: None", "\nInput:", "Input:", "Solution: \n\n", "Solution:\n", "Solution: ", "\nNext request.", " Next request."]
    for key in message_keys:
        for word in keywords:
            if word in contents[key]["content"]:
                contents[key]["content"] = contents[key]["content"].replace(word, "")
    return contents

def convert_to_chat_format(contents, message_keys):
    conversation_dict = {"id": str(uuid.uuid4()), "conversations": []}
    user_message_keys = [
        key for key in message_keys
        if contents[key]['role_type'] == 'USER'
    ]
    assistant_message_keys = [
        key for key in message_keys
        if contents[key]['role_type'] == 'ASSISTANT'
    ]
    for user_key, assistant_key in zip(user_message_keys, assistant_message_keys):
        conversation_dict["conversations"].append({"from": "human", "value": contents[user_key]["content"]})
        conversation_dict["conversations"].append({"from": "gpt", "value": contents[assistant_key]["content"]})

    if len(user_message_keys) > len(assistant_message_keys):
        conversation_dict["conversations"].append({"from": "human", "value": contents[user_message_keys[-1]]["content"]})
    if len(user_message_keys) < len(assistant_message_keys):
        conversation_dict["conversations"].append({"from": "gpt", "value": contents[assistant_message_keys[-1]]["content"]})

    return conversation_dict

def convert_science_to_conversation(file_list):
    science_conv_dataset = []
    for file in tqdm(file_list):

        with open(file) as f:
            contents = json.load(f)
        
        science_conv_dataset.append({'id':str(uuid.uuid4()),
                                    'conversations':[{"from": "human", "value": contents["message_1"]},
                                                    {"from": "gpt", "value": contents["message_2"]}]})
    return science_conv_dataset

if __name__ == "__main__":

    download_directory = "datasets"
    dataset_directories = [x for x in os.listdir(download_directory) if os.path.isdir(os.path.join(download_directory,x))]

    conv = []
    for directory in dataset_directories:
        print(f'Processing {directory}...')
        if directory == "ai_society_chat" or directory == "code_chat":
            dataset_directory = os.path.join(download_directory, directory)
            file_list = glob.glob(os.path.join(dataset_directory,"*.json"))
            conv = conv + convert_filelist_to_dataset(file_list)
        else:
            dataset_directory = os.path.join(download_directory, directory)
            file_list = glob.glob(os.path.join(dataset_directory,"*.json"))
            conv = conv + convert_science_to_conversation(file_list)

    random.shuffle(conv)
    print('Saving camel datasets...')
    with open(os.path.join(download_directory,"camel_datasets_conv.json"), 'w') as f:
        json.dump(conv, f, indent=2)

