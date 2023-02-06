import argparse
import json
import os

"""
NOTE: Please run this script BEFORE building "en2{tgt}_alignment.json" and "{tgt}2canonical.json", where {tgt} is the target language!
This script is used to combine all translations of source language entities which share the same canonicalized forms.

Please organize your translations into the following format:
{
    'domain1': {
        'slot1': {
            'src_value1': ['tgt_translation1', 'tgt_translation2', ...],
            ...
        },
        ...
    },
    ...
}
"""


parser = argparse.ArgumentParser(
    description="reorganize the bilingual alignment file to make all tranlations map to the canonical values"
)

parser.add_argument("--value_alignment_path", type=str, help="path of (preliminary) bilingual value alignment file")
parser.add_argument("--src_canonical_path", type=str, help="path of canonical value mapping file in the source language")
parser.add_argument("--output_path", type=str, help="output path of organized bilingual alignment file")

args = parser.parse_args()
with open(args.value_alignment_path, "r") as f:
    value_alignment = json.load(f)
with open(args.src_canonical_path, "r") as f:
    src_canonical = json.load(f)

organized_value_alignment = {}
for d in value_alignment.keys():
    organized_value_alignment[d] = {}
    for s in value_alignment[d].keys():
        organized_value_alignment[d][s.lower()] = {}
        for src_val, tgt_val_list in value_alignment[d][s].items():
            d, s = d.lower(), s.lower()
            if src_val in src_canonical[d][s].keys():
                canonical_src_val = src_val
            elif any([src_val in src_canonical[d][s][v] for v in src_canonical[d][s].keys()]):
                canonical_src_val = src_val
            else:
                print(f"Cannot find canonical value for {src_val} in {d} domain, {s} slot")
            if canonical_src_val not in organized_value_alignment[d][s].keys():
                organized_value_alignment[d][s][canonical_src_val] = []
            organized_value_alignment[d][s][canonical_src_val].extend(tgt_val_list)
        # remove duplicate values
        organized_value_alignment[d][s] = {k: list(set(v)) for k, v in organized_value_alignment[d][s].items()}
with open(os.path.join(args.output_path, "alignment_manual.json"), "w") as f:
    json.dump(organized_value_alignment, f, ensure_ascii=False, indent=4)
