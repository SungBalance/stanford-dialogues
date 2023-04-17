from pprint import pprint
import json
import re

import pandas as pd

FIX_MODE=False

# load original translated dataset
dataset_pd = [pd.read_csv(f"./datasets/english_{dataset}_out_final_output.csv", index_col=False, delimiter="\t") for dataset in ["dev", "few_shot_train", "test"]]
dataset_pd = pd.concat(dataset_pd).drop('Unnamed: 0', axis=1)
print("dataset loaded")

def check_exist_dataset(string):
    try:
        filter_mask = dataset_pd['source'].str.contains(re.escape(string))
    except Exception as e:
        print(e)
        print(string)
    if filter_mask.sum() == 0:
        return None
    else:
        return dataset_pd.loc[filter_mask, ['source', 'target']].reset_index(drop=True).loc[0].to_dict()
    

# load alignmnet file and error data
with open(f"./kr2en_alignment.json") as f:
    value_alignment = json.load(f)
with open(f"./kr2canonical.json") as f:
    canonical = json.load(f)

def save():
    with open(f"./kr2en_alignment.json", "w") as f:
        json.dump(value_alignment, f, ensure_ascii=False, indent=4)
    with open(f"./kr2canonical.json", "w") as f:
        json.dump(canonical, f, ensure_ascii=False, indent=4)

error_pd = pd.read_csv("./value_err.csv", index_col=False).dropna()
error_pd = error_pd[["domain", "src_slot", "src_value"]]
error_count = len(error_pd)
error_dic = {k: g.pivot_table(index='src_slot', values='src_value',aggfunc=list).to_dict()['src_value'] for k, g in error_pd.groupby('domain')}


# search and fix
missing_count = 0
already_count = 0
print(f"value_domains: {value_alignment.keys()}")
print(f"error_domains: {error_dic.keys()}")
for domain, error_slots in error_dic.items():
    print(f"\n--------- DOMAIN {domain} ---------")
    for slot, error_values in error_slots.items():
        print(f"\n:: DOMAIN {domain} SLOT {slot}")
        slot = slot.replace("_", " ")
        for error_value in error_values:
            if error_value in value_alignment[domain][slot]:
                already_count += 1
                continue
            example = check_exist_dataset(error_value)
            if example is not None:
                print('\033[93m'+f"\n:::: example\t:: {example['source']}"+'\033[0m')
                print('\033[93m'+f":::: example\t:: {example['target']}"+'\033[0m')
            else:
                print('\033[96m'+f"\n:::: noti\t:: the value '{error_value}' is not in dataset."+'\033[0m')
                missing_count += 1

            if FIX_MODE:
                not_pass = True
                while not_pass:
                    tgt_value = input(f":::: type\t:: '{error_value}' :: if type nothing, skip this. s for save:")
                    if not tgt_value:
                        not_pass = False
                        continue
                    if tgt_value == "s":
                        print(f":::: save !")
                        save()
                    else:
                        value_alignment[domain][slot][error_value] = tgt_value
                        if tgt_value in canonical[domain][slot]:
                            canonical[domain][slot][tgt_value] = [canonical[domain][slot][tgt_value], tgt_value]
                        else:
                            canonical[domain][slot][tgt_value] = tgt_value
                        
                        not_pass = False

print(f"{missing_count} in {error_count} is not in dataset already.")
print(already_count)