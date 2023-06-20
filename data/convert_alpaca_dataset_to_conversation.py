import json
import os
import uuid

from tqdm.auto import tqdm

if __name__ == "__main__":

  download_directory = "datasets"
  dataset_path = os.path.join(download_directory, "alpaca_data.json")

  if os.path.exists(dataset_path):
    with open(dataset_path) as f:
      alpaca_dataset = json.load(f)

    alpaca_conv_dataset = []

    for item in tqdm(alpaca_dataset):
        alpaca_conv_dataset.append({"id":str(uuid.uuid4()),
                                    "conversations":[{"from": "human", "value": item["instruction"] + " " + item["input"]},
                                                    {"from": "gpt", "value": item["output"]}]})

    with open(os.path.join(download_directory, "alpaca_data_conv.json"), "w") as f:
        json.dump(alpaca_conv_dataset, f, indent=2)
    else:
      print(f"{dataset_path} does not exist.")