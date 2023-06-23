import argparse
import glob
import json
import os
import shutil
import uuid

import pandas as pd
from tqdm.auto import tqdm

from .clean_sharegpt import clean_html_all
from .optional_clean import filter_language

DATASETS = {"ai_society": "ai_society_chat", 
            "code": "code_chat", 
            "math": "math", 
            "physics": "physics", 
            "chemistry": "chemistry", 
            "biology": "biology",
            "alpaca": "train-00000-of-00001-a09b74b3ef9c3b56.parquet",
            "sharegpt": ["sg_90k_part1.json", "sg_90k_part2.json"]}

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

def convert_alpaca_to_conv_dataset(dataset, dataset_directory):

    alpaca_dataset = pd.read_parquet(os.path.join(dataset_directory, "data", DATASETS[dataset]), engine='pyarrow')
    alpaca_conv_dataset = []

    for instruction, input, output in tqdm(zip(alpaca_dataset["instruction"], alpaca_dataset["input"], alpaca_dataset["output"])):
        alpaca_conv_dataset.append({"id":str(uuid.uuid4()),
        "conversations":[{"from": "human", "value": instruction + " " + input}, 
                         {"from": "gpt", "value": output}]})

    with open(os.path.join(dataset_directory, f"{dataset}_conv.json"), "w") as f:
        json.dump(alpaca_conv_dataset, f, indent=2)

    if os.path.exists(os.path.join(dataset_directory, "data")):
        shutil.rmtree(os.path.join(dataset_directory, "data"))


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

def convert_chat_to_dataset(dataset, dataset_directory):
    file_list = glob.glob(os.path.join(dataset_directory,"*.json"))
    conv_dataset = []
    for file in tqdm(file_list):
        with open(file) as f:
            contents = json.load(f)
        contents = {k.lower(): v for k, v in contents.items()}
        message_keys = [key for key in contents.keys() if 'message_' in key]
        filtered_content = filter_keywords(contents, message_keys)
        conv_dict = convert_to_chat_format(filtered_content, message_keys)
        conv_dataset = conv_dataset + [conv_dict]

    with open(os.path.join(os.path.dirname(dataset_directory), f"{dataset}_conv.json"), "w") as f:
        json.dump(conv_dataset, f, indent=2)

def convert_science_dataset(dataset, dataset_directory):
    file_list = glob.glob(os.path.join(dataset_directory,"*.json"))
    science_conv_dataset = []
    for file in tqdm(file_list):

        with open(file) as f:
            contents = json.load(f)
        
        science_conv_dataset.append({'id':str(uuid.uuid4()),
                                    'conversations':[{"from": "human", "value": contents["message_1"]},
                                                    {"from": "gpt", "value": contents["message_2"]}]})
    
    with open(os.path.join(os.path.dirname(dataset_directory), f"{dataset}_conv.json"), "w") as f:
        json.dump(science_conv_dataset, f, indent=2)

def convert_sharegpt_to_conv_dataset(dataset, dataset_directory, keep_lang="en"):
    sharegpt = []
    for file in DATASETS[dataset]:
        # Merge ShareGPT files
        with open(os.path.join(dataset_directory, "HTML_cleaned_raw_dataset", file)) as f:
            sharegpt = sharegpt + json.load(f)
    
    print("Converting ShareGPT HTML to markdown")
    sharegpt = clean_html_all(sharegpt)
    print("Keeping English language only...")
    sharegpt = filter_language(sharegpt, keep_lang="en")

    # Save merged ShareGPT files
    with open(os.path.join(dataset_directory, "sharegpt_conv.json"), "w") as f:
        json.dump(sharegpt, f, indent=2)
        
    # Remove HTML_cleaned_raw_dataset directory
    if os.path.exists(os.path.join(dataset_directory, "HTML_cleaned_raw_dataset")):
        shutil.rmtree(os.path.join(dataset_directory, "HTML_cleaned_raw_dataset"))
    
def process_dataset(dataset):
    if dataset == "ai_society" or dataset == "code":
        dataset_directory = os.path.join(args.download_directory, DATASETS[dataset])
        convert_chat_to_dataset(dataset, dataset_directory)
    elif dataset in ["math", "physics", "chemistry", "biology"]:
        dataset_directory = os.path.join(args.download_directory, DATASETS[dataset])
        convert_science_dataset(dataset, dataset_directory)
    elif dataset == "alpaca":
        dataset_directory = os.path.join(args.download_directory)
        convert_alpaca_to_conv_dataset(dataset, dataset_directory)
    elif dataset == "sharegpt":
        dataset_directory = os.path.join(args.download_directory)
        convert_sharegpt_to_conv_dataset(dataset, dataset_directory)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_directory", type=str, default="datasets")
    parser.add_argument("--datasets", nargs='+', choices=list(DATASETS.keys())+["all"], type=str)

    args = parser.parse_args()
    if args.datasets == ["all"]:
        args.datasets = DATASETS.keys()

    for dataset in args.datasets:
        print(f"Processing {dataset}...")
        process_dataset(dataset)

