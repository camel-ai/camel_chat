import argparse
import glob
import json
import os
import random

from .split_long_conversation import split_long_conversations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-directory", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="dataset.json")
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    new_content = []
    files = glob.glob(os.path.join(args.in_directory, "*_conv.json"))

    for file in files:
        content = json.load(open(file, "r"))
        print(f"{file} has {len(content)} conversations.")
        new_content.extend(content)
    
    
    new_content = split_long_conversations(new_content, args.tokenizer_path, max_length=args.max_length)
    random.shuffle(new_content)
    print(f"Saving to {os.path.join(args.in_directory, args.out_file)}")
    json.dump(new_content, open(os.path.join(args.in_directory, args.out_file), "w"), indent=2)
