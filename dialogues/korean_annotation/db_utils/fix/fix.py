from pprint import pprint
import json
import re

import pandas as pd

FIX_MODE=True

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

def check_startwith(text_dic, string):
    for key in text_dic.keys():
        if error_value.startswith(f"{key} "):
            split = error_value.split(f'{key} ')[1]
            print(split)
            tgt_value = f"{text_dic[key]} {split}"
            print(tgt_value)
            return tgt_value
    return None

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


# new_error_dic = {}
# for (key, value) in error_dic.items():
#    # Check if key is even then add pair to new dictionary
#    if key in ['movie']:
#        new_error_dic[key] = value
# error_dic = new_error_dic


for domain_idx, (domain, error_slots) in enumerate(error_dic.items()):
    print(f"\n--------- DOMAIN {domain} ({domain_idx+1}/{len(error_dic.keys())}) ---------")
    for slot_idx, (slot, error_values) in enumerate(error_slots.items()):
        print(f"\n:: DOMAIN {domain} ({domain_idx+1}/{len(error_dic.keys())}) SLOT {slot} ({slot_idx+1}/{len(error_slots.keys())})")
        slot = slot.replace("_", " ")
        for value_idx, error_value in enumerate(error_values):
            
            if error_value in value_alignment[domain][slot]:
                already_count += 1
                IN_DATASET_VALUES.append({
                    'domain': domain,
                    'slot': slot,
                    'source_value': error_value
                    })
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

                # continue
                similar_set = eval(error_pd.loc[error_pd['src_value']==error_value, 'Recomm'].to_numpy()[0])
                if similar_set[0] in value_alignment[domain][slot]:
                    print('\033[96m'+f"\n:::: similar\t:: first similar:: '{similar_set[0]}'"+'\033[0m')
                    print('\033[96m'+f":::: similar\t:: first similar:: '{value_alignment[domain][slot][similar_set[0]]}'"+'\033[0m')
                if len(similar_set) > 1:
                    if similar_set[1] in value_alignment[domain][slot]:
                        print('\033[96m'+f"\n:::: similar\t:: second similar:: '{similar_set[1]}'"+'\033[0m')
                        print('\033[96m'+f":::: similar\t:: second similar:: '{value_alignment[domain][slot][similar_set[1]]}'"+'\033[0m')
                

            if FIX_MODE:
                not_pass = True
                while not_pass:
                    
                    if example is not None and example['source'] == error_value:
                        tgt_value = example['target']
                    
                    else:
                        startwith_dic = {
                            'Dell': '델',
                            'HP': 'HP',
                            'Acer': '에이서'
                        }
                        startwith = check_startwith(startwith_dic, error_value)
                        if startwith is not None:
                            print(startwith)

                        ev_list = error_value.split(" ")
                        if len(ev_list) == 17:
                            def get_minus(string):
                                if string[0] == "-":
                                    return f"영하 {string[1:]}"
                                else:
                                    return string
                            suggest = f"최고 기온은 섭씨 {get_minus(ev_list[5])}도 그리고 최저 기온은 섭씨 {get_minus(ev_list[14])}도입니다"
                            print(f":::: suggest\t:: {suggest}")
                        tgt_value = input(f":::: type fix\t:: ({value_idx+1}/{len(error_values)})'{error_value}' :: if type nothing, skip this. g for suggest, s for save:")
                    
                    if tgt_value == "g":
                        print(f":::: suggest ! :: {suggest}")
                        tgt_value = suggest
                    
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

save()
print(f"{missing_count} in {error_count} is not in dataset already.")
# print(already_count)

pd.DataFrame(IN_DATASET_VALUES).to_csv("./in_dataset_errors.csv")
pd.DataFrame(NOT_IN_DATASET_VALUES).to_csv("./not_in_dataset_errors.csv")