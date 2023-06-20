import argparse
import json
import random
from typing import Dict, Optional, Sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True, nargs="+")
    parser.add_argument("--out-file", type=str, default="merged.json")
    args = parser.parse_args()

    new_content = []
    for in_file in args.in_file:
        content = json.load(open(in_file, "r"))
        print(f"{in_file} has {len(content)} conversations.")
        new_content.extend(content)
    random.shuffle(new_content)
    print(f"Saving to {args.out_file}")
    json.dump(new_content, open(args.out_file, "w"), indent=2)
