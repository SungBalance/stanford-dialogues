python check.py \
    --new_data_path="./dataset_english_json" \
    --annotation_path="./dataset_csv"


python convert_char_level.py \
    --csv_path="./dataset_csv/english_dev_out_final_output.csv" \
    --data_folder="./dataset_english_json" \
    --output_folder="./dataset_korean_json" \
    --target_lang="korean"

python convert_char_level.py \
    --csv_path="./dataset_csv/english_test_out_final_output.csv" \
    --data_folder="./dataset_english_json" \
    --output_folder="./dataset_korean_json" \
    --target_lang="korean"

python convert_char_level.py \
    --csv_path="./dataset_csv/english_few_shot_train_out_final_output.csv" \
    --data_folder="./dataset_english_json" \
    --output_folder="./dataset_korean_json" \
    --target_lang="korean"



python extract_alignment.py \
    --data_path="./dataset_korean_json" \
    --output_path="./db_utils"

python ./db_utils/combine_translations.py \
    --value_alignment_path="./db_utils/preliminary_bilingual_alignment.json" \
    --src_canonical_path="./db_utils/en2canonical.json" \
    --output_path="./db_utils/output"
