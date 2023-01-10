import json
import argparse
import os
from collections import OrderedDict


def build_alignment(value, translation, canonical_mark):
    assert len(translation) >= 1, "translation should not be empty"
    if len(translation) == 1:
        return (
            {value: translation[0].replace(canonical_mark, "")},
            {translation[0].replace(canonical_mark, ""): translation[0].replace(canonical_mark, "")},
        )
    else:
        standard_translation = [item for item in translation if canonical_mark in item]
        assert (
            len(standard_translation) == 1
        ), f"only one standard translation for each source value is allowed, but got {len(standard_translation)} for {value}"
        other_translation = [item for item in translation if canonical_mark not in item]
        return (
            {value: standard_translation[0].replace(canonical_mark, "")},
            {standard_translation[0].replace(canonical_mark, ""): other_translation},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert manually selected translations to two mappings: ")
    parser.add_argument("--alignment_manual_path", type=str, help="path of manually annotated alignment file")
    parser.add_argument("--output_path", type=str, help="path of output file")
    parser.add_argument("--lang", type=str, help="language")
    parser.add_argument("--canonical_mark", type=str, default="#", help="mark of canonical values")
    parser.add_argument("--incorrect_mark", type=str, default="@", help="mark of incorrect values")
    args = parser.parse_args()
    with open(args.alignment_manual_path, "r") as f:
        alignment = json.load(f)
    # check value with multiple or without standard translation
    for domain in alignment.keys():
        for slot in alignment[domain].keys():
            for value, translation in alignment[domain][slot].items():
                if len(translation) > 1 and args.incorrect_mark not in value:
                    if len([True for item in translation if args.canonical_mark in item]) != 1:
                        print(domain, slot, value)

    # get filtered alignment
    standard_translation = OrderedDict()
    standard_translation_mapping = OrderedDict()
    for domain in alignment.keys():
        standard_translation[domain.lower()] = OrderedDict()
        standard_translation_mapping[domain.lower()] = OrderedDict()
        for slot in alignment[domain].keys():
            standard_translation[domain.lower()][slot.lower()] = OrderedDict()
            standard_translation_mapping[domain.lower()][slot.lower()] = OrderedDict()
            for value, translation in alignment[domain][slot].items():
                if args.incorrect_mark not in value:
                    filtered_translation = [item for item in translation if args.incorrect_mark not in item]
                    translation_and_mapping = build_alignment(value, filtered_translation, args.canonical_mark)
                    standard_translation[domain.lower()][slot.lower()].update(translation_and_mapping[0])
                    standard_translation_mapping[domain.lower()][slot.lower()].update(translation_and_mapping[1])

    with open(os.path.join(args.output_path, f"{args.lang}2en_alignment.json"), "w") as f:
        json.dump(standard_translation, f, ensure_ascii=False, indent=4)
    with open(os.path.join(args.output_path, f"{args.lang}2canonical.json"), "w") as f:
        json.dump(standard_translation_mapping, f, ensure_ascii=False, indent=4)
