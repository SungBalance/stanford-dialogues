import argparse
import json
import os
from pathlib import Path
from contextlib import ExitStack
import collections


def read_json_files_in_folder(path, exclude_list=None):
    json_filename = [path + "/" + filename for filename in os.listdir(path) if ".json" in filename]
    if exclude_list is not None:
        for exclude_item in exclude_list:
            json_filename = [file for file in json_filename if exclude_item not in file]
    with ExitStack() as stack:
        files = [stack.enter_context(open(fname)) for fname in json_filename]
        data = {}
        for i in range(len(files)):
            data[Path(json_filename[i]).stem] = json.load(files[i], object_pairs_hook=collections.OrderedDict)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="extract alignment annotations from dataset and categorize them by domains and slots"
    )
    parser.add_argument("--data_path", type=str, help="path of folder containing dataset files")
    parser.add_argument("--output_path", type=str, help="output folder of (preliminary) bilingual value alignment file")
    args = parser.parse_args()

    # load dataset
    alignment = {}
    data = read_json_files_in_folder(args.data_path)
    for split in data.keys():
        for dialogue in data[split]:
            for turn in dialogue["dialogue"]:
                for role in ["user", "system"]:
                    for ds in turn[f"{role}_utterance"][1].keys():
                        d, s = ds.split("-")
                        if d not in alignment.keys():
                            alignment[d] = {}
                        if s not in alignment[d].keys():
                            alignment[d][s] = {}
                        for v in turn[f"{role}_utterance"][1][ds].keys():
                            for tgt_v, src_v in turn["alignment"].items():
                                if v == tgt_v:
                                    if isinstance(src_v, list):
                                        for src_v_item in src_v:
                                            if src_v_item not in alignment[d][s].keys():
                                                alignment[d][s][src_v_item] = []
                                            alignment[d][s][src_v_item].extend([tgt_v])
                                    elif isinstance(src_v, str):
                                        if src_v not in alignment[d][s].keys():
                                            alignment[d][s][src_v] = []
                                        alignment[d][s][src_v].extend([tgt_v])
                                    else:
                                        raise TypeError(f"source value {src_v} should be either list or str type")
    for d in alignment.keys():
        for s in alignment[d].keys():
            for v in alignment[d][s].keys():
                alignment[d][s][v] = list(set(alignment[d][s][v]))
    with open(os.path.join(args.output_path, "preliminary_bilingual_alignment.json"), "w") as f:
        json.dump(alignment, f, ensure_ascii=False, indent=4)
