import argparse
import logging
import json
import os
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple
from collections import OrderedDict

"""
What does this script do?
1. delete redundant/incorrent annotations
2. if translated utterances are revised, the corresponding annotations in csv files will be deleted
3. check if every source entity has a corresponding target entity (including missing and newly introduced entities)

NOTE: please solve all warnings to stop error propogation!

If you have any question about this script, please contact Tianhao Shen (thshen@tju.edu.cn).
"""


def build_id_to_file_mapping(data_path: str):
    dialogue_id_to_file = {}
    all_dialogue_data = {}
    converted_id = OrderedDict()

    for file in os.listdir(data_path):
        if file.endswith(".json"):
            with open(os.path.join(data_path, file), "r") as f:
                data = json.load(f)
                id_cnt = 1
                for dial in data:
                    dialogue_id_to_file[dial["dialogue_id"]] = file
                    all_dialogue_data[dial["dialogue_id"]] = dial
                    for turn in dial["dialogue"]:
                        for role in ["user", "system"]:
                            converted_id[f"{dial['dialogue_id']}->{turn['turn_id']}->{role}"] = id_cnt
                            id_cnt += 1
    return dialogue_id_to_file, all_dialogue_data, converted_id


def build_file_to_id_mapping(id_to_file_mapping: dict):
    file_to_id_mapping = {}
    for id, file in id_to_file_mapping.items():
        if file not in file_to_id_mapping:
            file_to_id_mapping[file] = []
        file_to_id_mapping[file].append(id)
    return file_to_id_mapping


def check_nested_type(iterable, tp):
    return all(isinstance(item, tp) for item in iterable)


def find_value(l: list, val) -> Tuple[bool, Optional[list]]:
    """
    find a value in list with somewhat fuzzy match
    inputs:
        case #1: val is str and l is list(str)
        case #2: val is list(int) and l is list(list(int))
    outputs:
        1. match status (True/False)
        2. location of matched values (return empty list if not found)
    """
    assert (isinstance(val, str) and check_nested_type(l, str)) or (
        isinstance(val, list) and check_nested_type(l, list) and all([check_nested_type(item, int) for item in l])
    )
    # 1. exact match: for case #1 and #2
    if val in l:
        match_loc = [idx for idx, item in enumerate(l) if item == val]
        return True, match_loc
    # 2. case insensitive match and fuzzy match (match if val is a substring of any str in list or vice versa): for case #1
    elif isinstance(val, str) and check_nested_type(l, str):
        val = val.lower()
        l = list(map(str.lower, l))
        # word length should be the same for fuzzy match
        match_loc = [
            idx for idx, item in enumerate(l) if len(val.split()) == len(item.split()) and (item in val or val in item)
        ]
        return (True, match_loc) if match_loc else (False, [])
    else:
        return False, []


def del_value(df, idx, dial_id, turn_id, role):
    assert isinstance(idx, int) or (isinstance(idx, list) and check_nested_type(idx, int))
    if isinstance(idx, int):
        idx = [idx]
    idx = list(set(idx))
    for field in [
        "source_entity",
        "target_entity",
        "source_span",
        "target_span",
        "source_word_span",
        "target_word_span",
    ]:
        if field in df.columns:
            orig_field_v = eval(
                df.loc[
                    (df["dialogue_id"] == dial_id) & (df["turn_id"] == turn_id) & (df["utterance_type"] == role), field
                ].values[0]
            )
            df.loc[
                (df["dialogue_id"] == dial_id) & (df["turn_id"] == turn_id) & (df["utterance_type"] == role), field
            ] = str([orig_field_v[item_idx] for item_idx in range(len(orig_field_v)) if item_idx not in idx])
    # refresh the selection
    df_item = df.loc[(df["dialogue_id"] == dial_id) & (df["turn_id"] == turn_id) & (df["utterance_type"] == role)]
    return df, df_item


def convert_single_span(utt, word_span):
    # convert single word span to char span
    utt = utt + " "
    char_span = [0, 0]
    word_idx = 0
    start_flag, end_flag = False, False
    for ch_idx in range(len(utt)):
        if word_idx == word_span[0] and not start_flag:
            char_span[0] = ch_idx
            start_flag = True
        if utt[ch_idx] == " ":
            word_idx += 1
            if word_idx == word_span[1] and not end_flag:
                char_span[1] = ch_idx
                end_flag = True
        if start_flag and end_flag:
            return char_span
    assert start_flag and end_flag


def convert_span(utt: str, word_span_list: list):
    # convert a word span (list) to a char span (list)
    if check_nested_type(word_span_list, int):
        return convert_single_span(utt, word_span_list)
    elif check_nested_type(word_span_list, list):
        return [convert_single_span(utt, word_span) for word_span in word_span_list]
    else:
        raise TypeError("items in word_span_list should be either int or list")


def extract_value_and_span(utt_anno):
    turn_v = []
    turn_v_span = []
    for v_info in utt_anno.values():
        for v, v_span in v_info.items():
            turn_v.append(v)
            turn_v_span += v_span
    return turn_v, turn_v_span


def get_diff(new_data, csv_path, csv_file, id_to_file, file_to_id, converted_id):
    # read the annotation csv file
    csv_anno = pd.read_csv(os.path.join(csv_path, csv_file), sep="\t")
    csv_anno.rename(columns={"Unnamed: 0": ""}, inplace=True)
    csv_dial_id_list = list(csv_anno["dialogue_id"].unique())
    for split_dial_ids in file_to_id.values():
        if csv_dial_id_list in split_dial_ids:
            dial_id_diff = set(split_dial_ids) - set(csv_dial_id_list)
            if dial_id_diff:
                logging.warning(
                    f"Missing dialogue ids in {csv_file}: {list(dial_id_diff)}, please annotate these dialogues!"
                )
    for dial_id, new_dial in tqdm(new_data.items()):
        for new_turn in new_dial["dialogue"]:
            turn_id = new_turn["turn_id"]
            for role in ["user", "system"]:
                anno_id = f"{dial_id}->{turn_id}->{role}"
                utt, utt_anno = new_turn[f"{role}_utterance"]
                turn_v, turn_v_span = extract_value_and_span(utt_anno)
                csv_anno_item = csv_anno.loc[
                    (csv_anno["dialogue_id"] == dial_id)
                    & (csv_anno["turn_id"] == turn_id)
                    & (csv_anno["utterance_type"] == role)
                ]
                if not csv_anno_item.empty:
                    if csv_anno_item["source"].values[0] != utt:
                        # drop the whole utterance for re-annotation
                        csv_anno = csv_anno.loc[
                            (csv_anno["dialogue_id"] != dial_id)
                            | (csv_anno["turn_id"] != turn_id)
                            | (csv_anno["utterance_type"] != role)
                        ]
                        logging.warning(
                            f"Please reannotate {id_to_file[dial_id]}->{anno_id} (#{converted_id[anno_id]}) in annotation tool."
                        )
                    else:
                        idx_to_del = []
                        for csv_v_idx in range(len(eval(csv_anno_item["source_entity"].values[0]))):
                            csv_v = eval(csv_anno_item["source_entity"].values[0])[csv_v_idx]
                            csv_v_char_span = eval(csv_anno_item["source_span"].values[0])[csv_v_idx]
                            csv_v_word_span = eval(csv_anno_item["source_word_span"].values[0])[csv_v_idx]
                            if not find_value(turn_v, csv_v)[0]:
                                # delete old values in the csv annotations if they are not in the new file
                                idx_to_del.append(csv_v_idx)
                                logging.info(
                                    f'Delete "{csv_v}" in {id_to_file[dial_id]}->{anno_id} (#{converted_id[anno_id]}) from csv annotations because the value "{csv_v}" is not in the new file.'
                                )
                                logging.debug(f"csv_v: {csv_v}, turn_v: {turn_v}")
                            if not (
                                find_value(turn_v_span, csv_v_char_span)[0]
                                or find_value(turn_v_span, csv_v_word_span)[0]
                            ):
                                # delete old values in the csv annotations if their spans are not in the new file
                                idx_to_del.append(csv_v_idx)
                                logging.info(
                                    f'Delete "{csv_v}" in {id_to_file[dial_id]}->{anno_id} (#{converted_id[anno_id]}) from csv annotations because the span of the value "{csv_v}" is not in the new file.'
                                )
                                logging.debug(
                                    f"csv_v: {csv_v}, turn_v: {turn_v}, csv_v: {turn_v_span}, csv_v_char_span: {csv_v_char_span}, csv_v_word_span: {csv_v_word_span}"
                                )
                        if idx_to_del:
                            logging.debug(f"csv_anno_item before deletion: {csv_anno_item}")
                            csv_anno, csv_anno_item = del_value(csv_anno, idx_to_del, dial_id, turn_id, role)
                            logging.debug(f"csv_anno_item after deletion: {csv_anno_item}")
                            idx_to_del.clear()
                        for v_info in utt_anno.values():
                            for v, v_span in v_info.items():
                                # new value is not in the csv file
                                if not find_value(eval(csv_anno_item["source_entity"].values[0]), v)[0]:
                                    logging.warning(
                                        f'Please annotate missing target entity for source entity "{v}" at {v_span} (char span: {convert_span(utt, v_span)}) in {id_to_file[dial_id]}->{anno_id} (#{converted_id[anno_id]}) in annotation tool.'
                                    )
                                    logging.debug(
                                        f"v: {v}, source_entity: {eval(csv_anno_item['source_entity'].values[0])}"
                                    )
                                else:
                                    # new value is in the csv file, but the span is different
                                    if not check_nested_type(v_span, list):
                                        v_span = [v_span]
                                    for span in v_span:
                                        if not (
                                            find_value(eval(csv_anno_item["source_span"].values[0]), span)[0]
                                            or find_value(eval(csv_anno_item["source_word_span"].values[0]), span)[0]
                                        ):
                                            # delete the old span
                                            idx_to_del += find_value(eval(csv_anno_item["source_entity"].values[0]), v)[
                                                1
                                            ]
                                            logging.debug(
                                                f"v: {v}, source_entity: {eval(csv_anno_item['source_entity'].values[0])}, idx_to_del: {idx_to_del}"
                                            )
                                            logging.warning(
                                                f'Please reannotate target entity span of source entity "{v}" at {span} (char span: {convert_span(utt, span)}) in {id_to_file[dial_id]}->{anno_id} (#{converted_id[anno_id]}) in annotation tool.'
                                            )
                        if idx_to_del:
                            logging.debug(f"csv_anno_item before deletion: {csv_anno_item}")
                            csv_anno, csv_anno_item = del_value(csv_anno, idx_to_del, dial_id, turn_id, role)
                            logging.debug(f"csv_anno_item after deletion: {csv_anno_item}")
                            idx_to_del.clear()
    orig_csv_anno = pd.read_csv(os.path.join(csv_path, csv_file), sep="\t")
    csv_diff = pd.concat([csv_anno, orig_csv_anno]).drop_duplicates(keep=False)
    return csv_anno, csv_diff


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--new_data_path", type=str, required=True, help="Path of the folder containing the new dataset")
    parser.add_argument("--annotation_path", type=str, required=True, help="Path to the annotation csv file")
    args = parser.parse_args()

    # build mapping between dialogue ids and the files they belong to
    new_dialogue_id_to_file, all_new_dialogue_data, converted_id = build_id_to_file_mapping(args.new_data_path)
    new_dialogue_file_to_id = build_file_to_id_mapping(new_dialogue_id_to_file)

    # get the differences of word spans and alignment annotations between the previous version and the current version of the dataset
    for file in os.listdir(args.annotation_path):
        if file.endswith(".csv") and (not file.startswith("new")):
            # parse the annotation diff and update the annotation csv file
            csv_anno, csv_diff = get_diff(
                new_data=all_new_dialogue_data,
                csv_path=args.annotation_path,
                csv_file=file,
                id_to_file=new_dialogue_id_to_file,
                file_to_id=new_dialogue_file_to_id,
                converted_id=converted_id,
            )
            csv_anno.to_csv(os.path.join(args.annotation_path, f"./new/{file}"), sep="\t", index=False)
            csv_diff.to_csv(os.path.join(args.annotation_path, f"./diff/{file}"), sep="\t", index=False)
