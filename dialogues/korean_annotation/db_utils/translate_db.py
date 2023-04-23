# Translate databases to target languages
import argparse
import json
from pathlib import Path

import pandas as pd
from pprint import pprint

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
    "",
)

# Read bilingual slot name and value alignment (standard translation)
# Please note that the slot alignment file is a json file which shares the same format as the "slot_alignment" dict
# used in convert.py (i.e., {src_slot1: tgt_slot1, src_slot2: tgt_slot2, ...})
with open(f"{args.slot_alignment_path}") as f:
    slot_alignment = {k: v.lower() for k, v in json.load(f).items()}
with open(f"{args.value_alignment_path}") as f:
    value_alignment = json.load(f)

TGT_SLOT_TRANS = False
TGT_SLOT_UNDERBAR = True
def get_tgt_slot(src_slot):
    if TGT_SLOT_TRANS:
        tgt_slot = slot_alignment[src_slot]
    else:
        tgt_slot = src_slot

    tgt_slot.strip()

    if TGT_SLOT_UNDERBAR:
        return tgt_slot.replace(" ", "_")
    else:
        return tgt_slot.replace("_", " ")

# BDSL
for domain in domain_list:
    slot_list = list(value_alignment[domain].keys())
    for src_slot in slot_list:
        tgt_slot = get_tgt_slot(src_slot)
        value_alignment[domain][tgt_slot] = value_alignment[domain].pop(src_slot)

ERR_DIC = {
    'domain': [],
    'slot': [],
    'value': []
}

COUNT_LIST = []
ERR_ALIGN = {}

for domain in domain_list:
# for domain in ['pc']:
    tgt_db = []
    # Read the source language db
    with open(f"{args.src_db_path}/{domain}_{args.src_lang}.json", "r") as f:
        src_db = json.load(f)
        print(f"Load {len(src_db)} items in {domain} domain from {args.src_lang} database")
    
    # Map the corresponding slot and value to the target language
    for src_db_item in src_db:
        tgt_db_item = {}
        print("... in for src_db_item")
        for src_slot, src_value in src_db_item.items():
            tgt_slot = get_tgt_slot(src_slot)
            # print(f"\nsrc: {src_slot} - {src_value} | tgt: {tgt_slot}")
            if isinstance(src_value, list):
                try:
                    tgt_db_item[tgt_slot] = [value_alignment[domain][tgt_slot][option] for option in src_value]
                except KeyError as e:
                    error_value = e.args[0]
                    print(
                        f"Warning: missing value '{e}' of slot '{src_slot}' in source language {args.src_lang} not found in the bilingual value alignment file! Please add the corresponding translation to the alignment file and try again."
                    )
                    if error_value in domain_list:
                        ERR_DIC['domain'].append({
                            'src_domain': error_value,
                            'target_domains': list(value_alignment.keys())
                        })
                    elif error_value == tgt_slot:
                        ERR_DIC['slot'].append({
                            'domain': domain,
                            'src_slot': error_value,
                            'target_slots': list(value_alignment[domain].keys())
                        })
                    else:
                        ERR_DIC['value'].append({
                            'domain': domain,
                            'src_slot': src_slot,
                            'tgt_slot': tgt_slot,
                            'src_value': error_value,
                            'target_values': list(value_alignment[domain][tgt_slot].keys())
                        })
            else:
                src_value = str(src_value)
                try:
                    if src_value in value_alignment[domain][tgt_slot].keys():
                        tgt_db_item[tgt_slot] = value_alignment[domain][tgt_slot][src_value]
                    elif not src_value:
                        tgt_db_item[tgt_slot] = na_translation[1]
                    else:
                        print(
                            f"Warning: missing value '{src_value}' of slot '{src_slot}' in source language {args.src_lang} not found in the bilingual value alignment file! Please add the corresponding translation to the alignment file and try again."
                        )
                        ERR_DIC['value'].append({
                            'domain': domain,
                            'src_slot': src_slot,
                            'tgt_slot': tgt_slot,
                            'src_value': src_value,
                            'target_values': list(value_alignment[domain][tgt_slot].keys())
                        })
                except KeyError as e:
                    error_value = e.args[0]
                    if error_value in domain_list:
                        ERR_DIC['domain'].append({
                            'src_domain': error_value,
                            'target_domains': list(value_alignment.keys())
                        })
                    elif error_value == str(tgt_slot):
                        ERR_DIC['slot'].append({
                            'domain': domain,
                            'src_slot': error_value,
                            'target_slots': list(value_alignment[domain].keys())
                        })
                    else:
                        ERR_DIC['value'].append({
                            'domain': domain,
                            'src_slot': src_slot,
                            'tgt_slot': tgt_slot,
                            'src_value': error_value,
                            'target_values': list(value_alignment[domain][tgt_slot].keys())
                        })
        # Check integrity of translated db item
        if len(src_db_item) == len(tgt_db_item):
            tgt_db.append({k.replace(" ", "_"): v for k, v in tgt_db_item.items()})
    print(f"Finished translation from {args.src_lang} to {args.tgt_lang} in {domain} domain!")
    print(f"successful: {len(tgt_db)}, failed: {len(src_db) - len(tgt_db)}\n\n\n")

    COUNT_LIST.append({
        'domain': domain,
        "successful": len(tgt_db),
        "failed": len(src_db) - len(tgt_db),
        "original_count_from_src_db": len(src_db)
    })
    
    # Write the target language db
    tgt_db_path = Path(f"{args.tgt_db_path}")
    tgt_db_path.mkdir(exist_ok=True)
    with open(f"{args.tgt_db_path}/{domain}_{args.tgt_lang}.json", "w") as f:
        json.dump(tgt_db, f, ensure_ascii=False, indent=4)
    

    # pprint(src_db)
    # pprint(tgt_db)
    print(domain)

count_pd = pd.DataFrame(COUNT_LIST)
err_domain_pd = pd.DataFrame(ERR_DIC['domain'])
err_slot_pd = pd.DataFrame(ERR_DIC['slot'])
err_value_pd = pd.DataFrame(ERR_DIC['value'])

# err_domain_pd.drop_duplicates()
err_slot_pd = err_slot_pd.drop_duplicates(['src_slot'])
err_value_pd = err_value_pd.drop_duplicates(['domain', 'src_slot', 'tgt_slot', 'src_value'])
print("")
pprint(count_pd)
print("")
pprint(err_slot_pd.head())
print("")
pprint(err_value_pd.head())

import jellyfish

def similar(source, target_list):
    score_list = [jellyfish.jaro_distance(source, target) for target in target_list]
    score, sentence = zip(*sorted(zip(score_list, target_list)))
    return sentence[-2:]

err_value_pd['Recomm'] = err_value_pd.apply(lambda x: similar(x.src_value, x.target_values), axis=1)
err_value_pd = err_value_pd[['domain', 'src_slot', 'tgt_slot', 'src_value', 'Recomm', 'target_values']]
pprint(err_value_pd.head())
pprint("")

print(f"MISSING VALUES:: domain {len(err_domain_pd)}, slot {len(err_slot_pd)}, value {count_pd['failed'].sum()}/{count_pd['original_count_from_src_db'].sum()} = {round(count_pd['failed'].sum()/count_pd['original_count_from_src_db'].sum(),4)*100}%")

err_slot_pd.to_csv("./slot_err.csv")
err_value_pd.to_csv("./value_err.csv")
