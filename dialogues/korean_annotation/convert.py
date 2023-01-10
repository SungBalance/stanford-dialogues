import argparse
import collections
import copy
import json
import os
import pandas as pd

from tqdm import tqdm
from contextlib import ExitStack
from pathlib import Path

# TODO: goal translation

### source-target domain/slot name alignment, please adapt values to your own language
domain_alignment = {
    # source language: target language
    "Attraction": "Attraction",
    "Car": "Car",
    "Class": "Class",
    "Flight": "Flight",
    "General": "General",
    "Hospital": "Hospital",
    "Hotel": "Hotel",
    "Movie": "Movie",
    "Other": "Other",
    "PC": "PC",
    "Restaurant": "Restaurant",
    "TV": "TV",
    "Train": "Train",
    "Weather": "Weather",
}
slot_alignment = {
    # source language: target language
    "3.0T MRI": "3.0T MRI",
    "4WD": "4WD",
    "CPU": "CPU",
    "CPU model": "CPU model",
    "CT": "CT",
    "DSA": "DSA",
    "Douban score": "Douban score",
    "GPU category": "GPU category",
    "GPU model": "GPU model",
    "UV intensity": "UV intensity",
    "address": "address",
    "area": "area",
    "arrival time": "arrival time",
    "brand": "brand",
    "bus routes": "bus routes",
    "business hours": "business hours",
    "campus": "campus",
    "city": "city",
    "class cabin": "class cabin",
    "class number": "class number",
    "classification": "classification",
    "classroom": "classroom",
    "colour": "colour",
    "computer type": "computer type",
    "consumption": "consumption",
    "cruise control system": " cruise control system",
    "cuisine": "cuisine",
    "date": "date",
    "day": "day",
    "decade": "decade",
    "departure": "departure",
    "departure time": "departure time",
    "destination": "destination",
    "director": "director",
    "dishes": "dishes",
    "duration": "duration",
    "end date": "end date",
    "end time": "end time",
    "episode length": "episode length",
    "episodes": "episodes",
    "features": "features",
    "film length": "film length",
    "flight information": "flight information",
    "fuel consumption": "fuel consumption",
    "game performance": "game performance",
    "general or specialized": "general or specialized",
    "grade": "grade",
    "hard disk capacity": "hard disk capacity",
    "heated seats": "heated seats",
    "hotel type": "hotel type",
    "hours": "hours",
    "hybrid": "hybrid",
    "key departments": "key departments",
    "level": "level",
    "localized": "localized",
    "memory capacity": "memory capacity",
    "metro station": "metro station",
    "name": "name",
    "name list": "name list",
    "number of seats": "number of seats",
    "opening hours": "opening hours",
    "operating system": "operating system",
    "parking": "parking",
    "parking assist system": "parking assist system",
    "per capita consumption": "per capita consumption",
    "phone": "phone",
    "phone number": "phone number",
    "power level": "power level",
    "premiere time": "premiere time",
    "price": "price",
    "pricerange": "pricerange",
    "product name": "product name",
    "production country or area": "production country or area",
    "public or private": "public or private",
    "punctuality rate": "punctuality rate",
    "registration time": "registration time",
    "release date": "release date",
    "room charge": "room charge",
    "room type": "room type",
    "score": "score",
    "screen size": "screen size",
    "seat type": "seat type",
    "series": "series",
    "service time": "service time",
    "size": "size",
    "standby time": "standby time",
    "star": "star",
    "start date": "start date",
    "start time": "start time",
    "subject": "subject",
    "tamperature": "tamperature",
    "teacher": "teacher",
    "the most suitable people": "the most suitable people",
    "ticket price": "ticket price",
    "time": "time",
    "times": "times",
    "title": "title",
    "train number": "train number",
    "type": "type",
    "usage": "usage",
    "ventilated seats": "ventilated seats",
    "weather condition": "weather condition",
    "weight": "weight",
    "wind": "wind",
}
domain_inverse_alignment = {v: k for k, v in domain_alignment.items()}
slot_inverse_alignment = {v: k for k, v in slot_alignment.items()}


def normalize_entity(entity):
    while entity[-1] in ",.;:!?":
        entity = entity[:-1]
    return entity


def find_value_in_belief_states(belief_state, value):
    occurence = []
    for k in belief_state.keys():
        if isinstance(belief_state[k], dict):
            for ds in belief_state[k].keys():
                if normalize_entity(value).lower() == belief_state[k][ds].lower():
                    d, s = ds.split("-")
                    occurence.append((k, d, s))
    return occurence


def find_value_in_actions(actions, value):
    occurence = []
    for action in actions:
        if action[3].lower() == normalize_entity(value).lower():
            occurence.append((action[1], action[2]))
    return occurence


def align(value, alignment):
    return alignment.get(value, value)


def reverse_dict(dict):
    reversed_dict = {}
    for k, v in dict.items():
        if v in reversed_dict.keys():
            reversed_dict[v].append(k)
        else:
            reversed_dict[v] = [k]
    for k, v in reversed_dict.items():
        if len(v) == 1:
            reversed_dict[k] = v[0]
    return reversed_dict


def build_utterance_from_annotation(annotation, source_turn, dialogue_id, role, domain_alignment, slot_alignment):
    utterance, utterance_annotation, value_alignment = [], {}, {}
    annotated_item = annotation.loc[
        (annotation["dialogue_id"] == dialogue_id)
        & (annotation["utterance_type"] == role)
        & (annotation["turn_id"] == source_turn["turn_id"])
    ]
    try:
        utterance.append(annotated_item["target"].values[0])
        try:
            source_entity = [entity.strip() for entity in eval(annotated_item["source_entity"].values[0])]
            target_entity = [entity.strip() for entity in eval(annotated_item["target_entity"].values[0])]
            assert len(source_entity) == len(
                target_entity
            ), f'Missing entity alignment detected. Please have a check! Location: \n dialogue {dialogue_id} \n -> turn #{source_turn["turn_id"]} \n -> {role} utterance'
            # add alignments
            value_alignment = {
                normalize_entity(source_entity[i]): normalize_entity(target_entity[i])
                for i in range(len(source_entity))
            }
        except IndexError:
            # Catch Exception: list index out of range
            return utterance + [{}], None
    except IndexError:
        # Catch Exception: index 0 is out of bounds for axis 0 with size 0
        return None, None

    for ds in source_turn[f"{role}_utterance"][1].keys():
        d, s = ds.split("-")
        target_d, target_s = align(d, domain_alignment), align(s, slot_alignment)
        utterance_annotation[f"{target_d}-{target_s}"] = {}
        for v in source_turn[f"{role}_utterance"][1][ds].keys():
            v_idx = [i for i, x in enumerate(source_entity) if normalize_entity(x).lower() == v.lower()]
            if v_idx:
                target_v = target_entity[v_idx[0]]
                target_v_span = [eval(annotated_item["target_word_span"].values[0])[i] for i in v_idx]
                target_v_span = [item for item in target_v_span if item not in [[0, 0], [-1, -1]]]
                if target_v_span:
                    utterance_annotation[f"{target_d}-{target_s}"][normalize_entity(target_v)] = target_v_span
                value_alignment[normalize_entity(v)] = normalize_entity(target_v)
        utterance_annotation = {k: v for k, v in utterance_annotation.items() if v}
    utterance.append(utterance_annotation)
    return utterance, value_alignment


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        help="path to csv file, e.g. 'english_dialog_sample_output.csv'",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        help="folder containing few-shot train, dev, and test json files, e.g. '.'",
    )
    parser.add_argument("--output_folder", type=str, help="folder of output json file, e.g. './converted'")
    parser.add_argument("--target_lang", type=str, help="target language, e.g. 'english'")
    args = parser.parse_args()

    annotation = pd.read_csv(args.csv_path, sep="\t")
    split = Path(args.csv_path).stem.strip("_output")
    continue_flag = False
    data = read_json_files_in_folder(args.data_folder)
    target_data = []
    for source_dialog in tqdm(data[split]):
        target_dialog = copy.deepcopy(source_dialog)
        # translate domains
        target_dialog["domains"] = [align(domain, domain_alignment) for domain in source_dialog["domains"]]
        # translate dialogue
        target_dialog["dialogue"] = []
        for turn_idx in range(len(source_dialog["dialogue"])):
            source_turn = source_dialog["dialogue"][turn_idx]
            target_turn = copy.deepcopy(source_turn)
            # translate turn domain
            target_turn["turn_domain"] = [align(domain, domain_alignment) for domain in source_turn["turn_domain"]]
            # build user utterance and system utterance from csv annotation
            target_turn["user_utterance"], user_value_alignment = build_utterance_from_annotation(
                annotation, source_turn, source_dialog["dialogue_id"], "user", domain_alignment, slot_alignment
            )
            target_turn["system_utterance"], system_value_alignment = build_utterance_from_annotation(
                annotation, source_turn, source_dialog["dialogue_id"], "system", domain_alignment, slot_alignment
            )
            if target_turn["user_utterance"] is not None and target_turn["system_utterance"] is not None:
                # translate belief state
                target_turn["belief_state"] = collections.OrderedDict()
                value_alignment = {**user_value_alignment, **system_value_alignment}
                for source_value in value_alignment.keys():
                    for k, d, s in find_value_in_belief_states(source_turn["belief_state"], source_value):
                        target_turn["belief_state"][k] = {}
                        target_d, target_s, target_v = (
                            align(d, domain_alignment),
                            align(s, slot_alignment),
                            align(source_value, value_alignment),
                        )
                        target_turn["belief_state"][k][f"{target_d}-{target_s}"] = target_v
                target_turn["belief_state"]["turn request"] = [
                    align(slot, slot_alignment) for slot in source_turn["belief_state"]["turn request"]
                ]
                # translate actions
                for role in ["user", "system"]:
                    target_turn[f"{role}_actions"] = []
                    for action in source_turn[f"{role}_actions"]:
                        target_turn[f"{role}_actions"].append(
                            [
                                action[0],
                                align(action[1], domain_alignment),
                                align(action[2], slot_alignment),
                                align(action[3], value_alignment),
                            ]
                        )
                target_turn["alignment"] = reverse_dict(value_alignment)
                target_dialog["dialogue"].append(target_turn)
            else:
                continue_flag = True
                continue
        target_data.append(target_dialog)
        if continue_flag:
            continue
    with open(os.path.join(args.output_folder, f"{args.target_lang}_{split}.json"), "w") as f:
        # only output non-empty dialogue sessions
        target_data = [dialog for dialog in target_data if dialog["dialogue"]]
        json.dump(target_data, f, ensure_ascii=False, indent=4)
