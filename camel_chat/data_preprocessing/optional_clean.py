import argparse
import json
import re

import polyglot
import pycld2
from polyglot.detect import Detector
from tqdm import tqdm


def skip(conv, keep_lang="all", skip_lang=None, reduce_rep=False):
    # Remove certain languages
    if keep_lang != "all" or skip_lang is not None:
        text = "\n".join([x["value"] for x in conv["conversations"]])
        try:
            lang_code = Detector(text).language.code
        except (pycld2.error, polyglot.detect.base.UnknownLanguage):
            lang_code = "unknown"

        if keep_lang != "all" and lang_code != keep_lang:
            return True

        if lang_code == skip_lang:
            return True

    # Remove repetitive numbers
    if reduce_rep:
        for sentence in conv["conversations"]:
            val = sentence["value"]
            sub = re.search(r"(\d)\1{8}", val)
            if sub is not None:
                return True

    return False

def filter_language(content, keep_lang="en", skip_lang=None, reduce_rep=False):
    new_content = []
    for conv in tqdm(content):
        if not skip(conv, keep_lang=keep_lang, skip_lang=skip_lang, reduce_rep=reduce_rep):
            new_content.append(conv)
    return new_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str)
    parser.add_argument(
        "--keep-lang",
        type=str,
        default="all",
        choices=["all", "en"],
        help="Only keep certain langauges.",
    )
    parser.add_argument("--skip-lang", type=str, help="Skip a specific language.")
    # NOTE: Be careful about reduce_rep which may remove some good data.
    # For example, addresses could have long consecutive 0's
    parser.add_argument("--reduce-rep", action="store_true")
    args = parser.parse_args()

    in_file = args.in_file
    out_file = args.out_file
    keep_lang = args.keep_lang
    skip_lang = args.skip_lang
    reduce_rep = args.reduce_rep
    assert keep_lang == "all" or skip_lang is None

    if out_file is None:
        out_file = "sharegpt_clean"
        if keep_lang != "all":
            out_file += "_" + keep_lang
        if skip_lang is not None:
            out_file += "_skip_" + skip_lang
        if reduce_rep:
            out_file += "_reduce_rep"
        out_file += ".json"

    content = json.load(open(in_file, "r"))
    num_conv = len(content)
    new_content = filter_language(content, keep_lang=args.keep_lang, skip_lang=args.skip_lang, reduce_rep=args.reduce_rep)
    print(f"return {len(new_content)} out of {len(content)}, start dump ...")
    json.dump(new_content, open(out_file, "w"), indent=2)
