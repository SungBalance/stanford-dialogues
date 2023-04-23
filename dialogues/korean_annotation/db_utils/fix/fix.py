from pprint import pprint
import json
import re

import pandas as pd

FIX_MODE=False

# load original translated dataset
dataset_pd = [pd.read_csv(f"./datasets/english_{dataset}_out_final_output.csv", index_col=False, delimiter="\t") for dataset in ["dev", "few_shot_train", "test"]]
dataset_pd = pd.concat(dataset_pd).drop('Unnamed: 0', axis=1)
print("dataset loaded")

def check_exist_dataset(domain, string):
    try:
        # 'attraction_goal_1-13_v2###8382'
        domain_dataset_pd = dataset_pd[dataset_pd['dialogue_id'].str.split('_').str[0] == domain.lower()]
        filter_mask = domain_dataset_pd['source'].str.contains(re.escape(string))
    except Exception as e:
        print(string)
        print(str(e))
    if filter_mask.sum() == 0:
        return None
    else:
        return domain_dataset_pd.loc[filter_mask, ['source', 'target']].reset_index(drop=True).loc[0].to_dict()
    

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
error_pd = error_pd[["domain", "src_slot", "src_value", "Recomm"]]
error_count = len(error_pd)
error_dic = {k: g.pivot_table(index='src_slot', values='src_value',aggfunc=list).to_dict()['src_value'] for k, g in error_pd.groupby('domain')}


# search and fix
missing_count = 0
already_count = 0
print(f"value_domains: {value_alignment.keys()}")
print(f"error_domains: {error_dic.keys()}")

NOT_IN_DATASET_VALUES = []
IN_DATASET_VALUES = []

for domain, error_slots in error_dic.items():
    print(f"\n--------- DOMAIN {domain} ---------")
    for slot, error_values in error_slots.items():
        print(f"\n:: DOMAIN {domain} SLOT {slot}")
        slot = slot.replace("_", " ")
        for error_value in error_values:
            
            if error_value in value_alignment[domain][slot]:
                already_count += 1
                continue
            
            example = check_exist_dataset(domain, error_value)
            
            if example is not None: # errors in dataset
                print('\033[93m'+f"\n:::: dataset\t:: {example['source']}"+'\033[0m')
                print('\033[93m'+f":::: dataset\t:: {example['target']}"+'\033[0m')

                IN_DATASET_VALUES.append({
                    'domain': domain,
                    'slot': slot,
                    'source_value': error_value
                    })

            else: # errors not in dataset
                NOT_IN_DATASET_VALUES.append({
                    'domain': domain,
                    'slot': slot,
                    'source_value': error_value
                    })
                missing_count += 1
                continue # remove this to fix

                similar_set = eval(error_pd.loc[error_pd['src_value']==error_value, 'Recomm'].to_numpy()[0])
                if similar_set[0] in value_alignment[domain][slot]:
                    print('\033[96m'+f"\n:::: similar\t:: first similar:: '{similar_set[0]}'"+'\033[0m')
                    print('\033[96m'+f":::: similar\t:: first similar:: '{value_alignment[domain][slot][similar_set[0]]}'"+'\033[0m')
                if similar_set[1] in value_alignment[domain][slot]:
                    print('\033[96m'+f"\n:::: similar\t:: second similar:: '{similar_set[1]}'"+'\033[0m')
                    print('\033[96m'+f":::: similar\t:: second similar:: '{value_alignment[domain][slot][similar_set[1]]}'"+'\033[0m')
                

            if FIX_MODE:
                not_pass = True
                while not_pass:
                    if example['source'] == error_value:
                        tgt_value = example['target']
                    else:
                        tgt_value = input(f":::: type fix\t:: '{error_value}' :: if type nothing, skip this. s for save:")
                    
                    if not tgt_value:
                        not_pass = False
                        continue
                    if tgt_value == "s":
                        print(f":::: save !")
                        save()
                    else:
                        value_alignment[domain][slot][error_value] = tgt_value
                        if tgt_value not in canonical[domain][slot]:
                            canonical[domain][slot][tgt_value] = tgt_value
                        
                        not_pass = False
                
                print("\n------------------")

print(f"{missing_count} in {error_count} is not in dataset already.")
print(already_count)

pd.DataFrame(IN_DATASET_VALUES).to_csv("./in_dataset_errors.csv")
pd.DataFrame(NOT_IN_DATASET_VALUES).to_csv("./not_in_dataset_errors.csv")