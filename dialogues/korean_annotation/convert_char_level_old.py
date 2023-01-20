import argparse
import collections
import copy
import json
import os
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# TODO: goal translation

### source-target domain/slot name alignment, please adapt values to your own language
domain_alignment = {
    # source language: target language
    'Attraction': '명소',
    'Car': '자동차',
    'Class': '수업',
    'Flight': '비행',
    'General': '일반',
    'Hospital': '병원',
    'Hotel': '호텔',
    'Movie': '영화',
    'Other': '기타',
    'PC': 'PC',
    'Restaurant': '식당',
    'TV': 'TV',
    'Train': '기차',
    'Weather': '날씨',
}
slot_alignment = {
    # source language: target language
    '3.0T MRI': '3.0T MRI',
    '4WD': '사륜구동',
    'CPU': 'CPU',
    'CPU model': 'CPU 모델',
    'CT': 'CT',
    'DSA': 'DSA',
    'Douban score': 'Douban 평점',
    'GPU category': 'GPU 유형',
    'GPU model': 'GPU 평점',
    'UV intensity': '자외선 강도',
    'address': '주소',
    'area': '지역',
    'arrival time': '도착시간',
    'brand': '상표',
    'bus routes': '버스 노선',
    'business hours': '업무 시간',
    'campus': '캠퍼스',
    'city': '도시',
    'class cabin': '객실 등급',
    'class number': '수업 번호',
    'classification': '분류',
    'classroom': '강의실',
    'colour': '색',
    'computer type': '컴퓨터 종류',
    'consumption': '소비',
    'cruise control system': '크루즈 컨트롤 시스템',
    'cuisine': '요리',
    'date': '날짜',
    'day': '요일',
    'decade': '10년',
    'departure': '출발',
    'departure time': '출발 시간',
    'destination': '목적지',
    'director': '감독',
    'dishes': '요리',
    'duration': '기간',
    'end date': '종료일',
    'end time': '종료 시간',
    'episode length': '회당 길이',
    'episodes': '에피소드',
    'features': '특징',
    'film length': '상영 길이',
    'flight information': '항공편 정보',
    'fuel consumption': '연료 소비량',
    'game performance': '게임 성능',
    'general or specialized': '종합 또는 전문',
    'grade': '성적',
    'hard disk capacity': '하드디스크 용량',
    'heated seats': '열선 시트',
    'hotel type': '호텔 종류',
    'hours': '시간',
    'hybrid': '하이브리드',
    'key departments': '주요 부서',
    'level': '단계',
    'localized': '현지화된',
    'memory capacity': '메모리 용량',
    'metro station': '지하철역',
    'name': '이름',
    'name list': '명단',
    'number of seats': '좌석수',
    'opening hours': '영업 시간',
    'operating system': '운영 체제',
    'parking': '주차',
    'parking assist system': '주차 보조 시스템',
    'per capita consumption': '일 인당 소비량',
    'phone': '전화',
    'phone number': '전화번호',
    'power level': '출력 수준',
    'premiere time': '첫 방영 시간',
    'price': '가격',
    'pricerange': '가격대',
    'product name': '상품명',
    'production country or area': '제작 국가 또는 제작 지역',
    'public or private': '공립 또는 사립',
    'punctuality rate': '정시율',
    'registration time': '등록 시간',
    'release date': '개봉일',
    'room charge': '객실 요금',
    'room type': '객실 타입',
    'score': '점수',
    'screen size': '화면 크기',
    'seat type': '좌석 종류',
    'series': '시리즈',
    'service time': '진료 시간',
    'size': '크기',
    'standby time': '대기 시간',
    'star': '주연',
    'start date': '시작일',
    'start time': '시작 시간',
    'subject': '주제',
    'tamperature': '기온',
    'teacher': '선생님',
    'the most suitable people': '가장 적당한 사람',
    'ticket price': '표 가격',
    'time': '시간',
    'times': '시간',
    'title': '제목',
    'train number': '기차 번호',
    'type': '종류',
    'usage': '사용',
    'ventilated seats': '통풍 시트',
    'weather condition': '기상 상황',
    'weight': '무게',
    'wind': '바람',
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
                normalize_entity(source_entity[i]): normalize_entity(target_entity[i]) for i in range(len(source_entity))
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
            # better support for char-level alignment
            v_idx = [
                i
                for i, x in enumerate(source_entity)
                if v.lower() in normalize_entity(x).lower() and len(v.split()) == len(x.split())
            ]
            if v_idx:
                target_v = target_entity[v_idx[0]]
                target_v_span = [eval(annotated_item["target_span"].values[0])[i] for i in v_idx]
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
