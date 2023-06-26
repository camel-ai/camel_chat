import argparse
import os

from huggingface_hub import snapshot_download


def dwonload_camel_models(model_name, download_directory):
    
    snapshot_download(repo_id=f"camel-ai/{model_name}", local_dir=os.path.join(download_directory, model_name), local_dir_use_symlinks=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["CAMEL-13B-Role-Playing-Data", 
                                                                          "CAMEL-13B-Combined-Data", 
                                                                          "CAMEL-33B-Combined-Data"])
    parser.add_argument("--download_directory", type=str, default="finetuned_models")
    args = parser.parse_args()

    dwonload_camel_models(args.model_name, args.download_directory)