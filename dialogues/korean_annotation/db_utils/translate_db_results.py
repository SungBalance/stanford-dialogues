# translate db_results
import argparse
import json
import pprint
from pathlib import Path
from tqdm import trange


def find_sub_list(sub_list, orig_list):
    results = []
    sub_list_len = len(sub_list)
    for idx in (i for i, e in enumerate(orig_list) if e == sub_list[0]):
        if " ".join(sub_list) in " ".join(orig_list[idx : idx + sub_list_len]):
            results.append([idx, idx + sub_list_len])
    return results


def normalize_dict(d):
    for k, v in d.items():
        if isinstance(v, bool) or isinstance(v, float) or isinstance(v, int):
            d[k] = str(v).lower()
        elif isinstance(v, str) and v.lower() in ["true", "false"]:
            d[k] = v.lower()
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str, help="source language")
    parser.add_argument("--tgt_lang", type=str, help="target language")
    parser.add_argument("--src_data_path", type=str, help="path of source data folder")
    parser.add_argument("--tgt_data_path", type=str, help="path of target data folder")
    parser.add_argument("--src_db_path", type=str, help="path of source database folder")
    parser.add_argument("--tgt_db_path", type=str, help="path of target database folder")
    parser.add_argument("--output_path", type=str, help="output path of data with db_results in the target language")
    parser.add_argument("--value_alignment_path", type=str, help="path of bilingual value alignment file")
    parser.add_argument("--debug", help="debug mode", action="store_true")

    args = parser.parse_args()

    domain_list = [
        "weather",
        "train",
        "pc",
        "movie",
        "class",
        "car",
        "restaurant",
        "hotel",
        "attraction",
        "flight",
        "hospital",
        "tv",
    ]

    # Translate the prefix of db_results: 'Database search results: the number of successful matches is '
    # to the target language. Please add your own translation to the second element of the tuple below.
    db_result_prefix = (
        # (source language, target language)
        "Database search results: the number of successful matches is ",
        "",
    )

    # load src and tgt data
    # please make sure the src and tgt data files have been renamed as "{lang}_{split}.json"
    # e.g. en_fewshot.json, en_valid.json, en_test.json
    src_data = []
    tgt_data = []
    split_list = ["fewshot", "valid", "test"]
    for split in split_list:
        with open(f"{args.src_data_path}/{args.src_lang}_{split}.json", "r") as f:
            src_data.append(json.load(f))
        with open(f"{args.tgt_data_path}/{args.tgt_lang}_{split}.json", "r") as f:
            tgt_data.append(json.load(f))
    # load src and tgt databases
    src_db = {}
    tgt_db = {}
    for domain in domain_list:
        with open(f"{args.src_db_path}/{domain}_{args.src_lang}.json", "r") as f:
            # normalize src db during loading
            src_db[domain] = [
                {k: (str(v) if str(v).lower() not in ["true", "false"] else str(v).lower()) for k, v in item.items()}
                for item in json.load(f)
            ]
            print(f"Load {len(src_db[domain])} items in {domain} domain from {args.src_lang} database")
        with open(f"{args.tgt_db_path}/{domain}_{args.tgt_lang}.json", "r") as f:
            tgt_db[domain] = json.load(f)
            print(f"Load {len(tgt_db[domain])} items in {domain} domain from {args.tgt_lang} database")
    # load bilingual value alignment file
    with open(args.value_alignment_path, "r") as f:
        value_alignment = json.load(f)

    total_set = set()
    match_set = set()
    not_match_set = set()

    # match each result in db_results with Chinese databases
    for split_idx in trange(len(src_data)):
        split = src_data[split_idx]
        for dialog_idx in trange(len(split)):
            dialog = split[dialog_idx]
            for turn_idx in range(len(dialog["dialogue"])):
                turn = dialog["dialogue"][turn_idx]
                tgt_db_results = []
                if "db_results" in turn.keys():
                    for src_db_result in turn["db_results"]:
                        if db_result_prefix[0] in src_db_result:
                            # translate the prefix of db_result
                            tgt_db_results.append(src_db_result.replace(db_result_prefix[0], db_result_prefix[1]))
                        else:
                            # turn dict string into dict
                            total_set.add(src_db_result)
                            src_db_result = eval(src_db_result.replace("true", "True").replace("false", "False"))
                            # convert True/False to 'true'/'false'
                            for k, v in src_db_result.items():
                                if isinstance(v, (bool, float, int, list)):
                                    src_db_result[k] = str(v).lower()

                            # locate result in the src databases
                            match_flag = False
                            for d in src_db.keys():
                                for item_idx in range(len(src_db[d])):
                                    src_db_item = src_db[d][item_idx]
                                    if src_db_item.items() == src_db_result.items():
                                        # match
                                        match_flag = True
                                        tgt_db_results.append(
                                            {k.replace(" ", "_"): v for k, v in tgt_db[d][item_idx].items()}
                                        )
                                        match_set.add(str(src_db_result))
                                        break
                            if not match_flag:
                                # not match
                                if args.debug:
                                    print(src_db_result)
                                not_match_set.add(str(src_db_result))
                if len(tgt_db_results) != len(turn["db_results"]):
                    print("not match")
                tgt_data[split_idx][dialog_idx]["dialogue"][turn_idx]["db_results"] = tgt_db_results
    print(len(not_match_set), len(match_set), len(total_set))
    if args.debug:
        pprint.pprint(not_match_set)

    output_path = Path(f"{args.output_path}")
    output_path.mkdir(exist_ok=True)
    for split_idx in range(len(tgt_data)):
        with open(f"{args.output_path}/{args.tgt_lang}_{split_list[split_idx]}.json", "w") as f:
            json.dump(tgt_data[split_idx], f, ensure_ascii=False, indent=4)
