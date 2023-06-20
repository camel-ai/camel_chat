import os
import tarfile
import zipfile

from huggingface_hub import hf_hub_download

download_directory = "datasets"

if not os.path.exists(download_directory):
    os.makedirs(download_directory)

# Download CAMEL datasets
hf_hub_download(repo_id="camel-ai/ai_society", repo_type="dataset", filename="ai_society_chat.tar.gz",
                local_dir=download_directory, local_dir_use_symlinks=False)

hf_hub_download(repo_id="camel-ai/code", repo_type="dataset", filename="code_chat.tar.gz",
                local_dir=download_directory, local_dir_use_symlinks=False)

hf_hub_download(repo_id="camel-ai/math", repo_type="dataset", filename="math50k.zip",
                  local_dir=download_directory, local_dir_use_symlinks=False)

hf_hub_download(repo_id="camel-ai/chemistry", repo_type="dataset", filename="chemistry.zip",
                local_dir=download_directory, local_dir_use_symlinks=False)

hf_hub_download(repo_id="camel-ai/biology", repo_type="dataset", filename="biology.zip",
                local_dir=download_directory, local_dir_use_symlinks=False)

hf_hub_download(repo_id="camel-ai/physics", repo_type="dataset", filename="physics.zip",
                local_dir=download_directory, local_dir_use_symlinks=False)

# Extract CAMEL datasets and remove compressed files
files = os.listdir(download_directory)

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

    