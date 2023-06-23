import argparse
import json
import os
import tarfile
import zipfile

from huggingface_hub import hf_hub_download

DATASETS = {"ai_society": "ai_society_chat.tar.gz", 
            "code": "code_chat.tar.gz", 
            "math": "math.zip", 
            "physics": "physics.zip", 
            "chemistry": "chemistry.zip", 
            "biology": "biology.zip",
            "sharegpt": ["sg_90k_part1.json", "sg_90k_part2.json"],
            "alpaca": "train-00000-of-00001-a09b74b3ef9c3b56.parquet"}

# Download datasets
def download_hf_dataset(dataset, download_directory):
    if dataset == "sharegpt":
        for file in DATASETS[dataset]:
            if os.path.exists(os.path.join(download_directory, "HTML_cleaned_raw_dataset", file)):
                continue
            hf_hub_download(repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered", repo_type="dataset", filename=file,
                            subfolder="HTML_cleaned_raw_dataset", local_dir=download_directory, local_dir_use_symlinks=False)
    elif dataset == "alpaca":
            if os.path.exists(os.path.join(download_directory, "data", DATASETS[dataset])):
                return
            hf_hub_download(repo_id="tatsu-lab/alpaca", repo_type="dataset", filename=DATASETS[dataset],
                            subfolder="data", local_dir=download_directory, local_dir_use_symlinks=False)
    else:
        if os.path.exists((os.path.join(download_directory, DATASETS[dataset]))):
            return
        hf_hub_download(repo_id=f"camel-ai/{dataset}", repo_type="dataset", filename=DATASETS[dataset],
                        local_dir=download_directory, local_dir_use_symlinks=False)

# Extract CAMEL datasets and remove compressed files
def unzip_datasets(download_directory):
    print("Extracting datasets...")
    files = [f for f in os.listdir(download_directory) if os.path.isfile(os.path.join(download_directory, f))]
    for file in files:
        file_path = os.path.join(download_directory, file)
        if tarfile.is_tarfile(file_path):
            with tarfile.open(file_path) as tar:
                tar.extractall(os.path.join("datasets", file.split('.')[0]))
            os.remove(file_path)
        elif zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join("datasets", file.split('.')[0]))
            os.remove(file_path)
        else: 
            pass

def download_dataset(dataset, download_directory):

    # try:
    download_hf_dataset(dataset, download_directory)
    # except:
    #     print(f"{dataset} could not be downloaded")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--download_directory", type=str, default="datasets")
    parser.add_argument("--datasets", nargs='+', choices=list(DATASETS.keys())+["all"], type=str)
    
    args = parser.parse_args()

    if not os.path.exists(args.download_directory):
        os.makedirs(args.download_directory)

    if args.datasets == ["all"]:
        args.datasets = list(DATASETS.keys())
    
    for dataset in args.datasets:
        download_dataset(dataset, args.download_directory)

    unzip_datasets(args.download_directory)
