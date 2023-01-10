# Translate databases to target languages
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Translate databases from a source language to a target language.")
parser.add_argument("--src_lang", type=str, help="source language")
parser.add_argument("--tgt_lang", type=str, help="target language")
parser.add_argument("--src_db_path", type=str, help="path of source database folder")
parser.add_argument("--tgt_db_path", type=str, help="path of target database folder")
parser.add_argument("--slot_alignment_path", type=str, help="path of bilingual slot alignment file")
parser.add_argument("--value_alignment_path", type=str, help="path of bilingual value alignment file")

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

# Translate special word "N/A" (i.e., not available) to the target language.
# Please add your own translation to the second element of the tuple below.
na_translation = (
    # (source language, target language)
    "N/A",
    ""
)

# Read bilingual slot name and value alignment (standard translation)
# Please note that the slot alignment file is a json file which shares the same format as the "slot_alignment" dict
# used in convert.py (i.e., {src_slot1: tgt_slot1, src_slot2: tgt_slot2, ...})
with open(f"{args.slot_alignment_path}") as f:
    slot_alignment = {k: v.lower() for k, v in json.load(f).items()}
with open(f"{args.value_alignment_path}") as f:
    value_alignment = json.load(f)

for domain in domain_list:
    tgt_db = []
    # Read the source language db
    with open(f"{args.src_db_path}/{domain}_{args.src_lang}.json", "r") as f:
        src_db = json.load(f)
        print(f"Load {len(src_db)} items in {domain} domain from {args.src_lang} database")

    # Map the corresponding slot and value to the target language
    for src_db_item in src_db:
        tgt_db_item = {}
        for src_slot, src_value in src_db_item.items():
            tgt_slot = slot_alignment[src_slot]
            if isinstance(src_value, list):
                tgt_db_item[tgt_slot] = [value_alignment[domain][tgt_slot][option] for option in src_value]
            else:
                src_value = str(src_value)
                if src_value in value_alignment[domain][tgt_slot].keys():
                    tgt_db_item[tgt_slot] = value_alignment[domain][tgt_slot][src_value]
                elif not src_value:
                    tgt_db_item[tgt_slot] = na_translation[1]
                else:
                    print(
                        f"Warning: missing value {src_value} of slot {src_slot} in source language {args.src_lang} not found in the bilingual value alignment file! Please add the corresponding translation to the alignment file and try again."
                    )
        # Check integrity of translated db item
        if len(src_db_item) == len(tgt_db_item):
            tgt_db.append({k.replace(" ", "_"): v for k, v in tgt_db_item.items()})
    print(f"Finished translation from {args.src_lang} to {args.tgt_lang} in {domain} domain!")
    print(f"successful: {len(tgt_db)}, failed: {len(src_db) - len(tgt_db)}")

    # Write the target language db
    tgt_db_path = Path(f"{args.tgt_db_path}")
    tgt_db_path.mkdir(exist_ok=True)
    with open(f"{args.tgt_db_path}/{domain}_{args.tgt_lang}.json", "w") as f:
        json.dump(tgt_db, f, ensure_ascii=False, indent=4)
