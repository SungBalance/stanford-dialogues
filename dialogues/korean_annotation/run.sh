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
    --output_path="./db_utils/outputs"

python ./db_utils/build_mappings.py \
    --alignment_manual_path="./db_utils/outputs/alignment_manual_fix.json" \
    --output_path="./db_utils/outputs" \
    --lang="kr" \
    --canonical_mark="#" \
    --incorrect_mark="@"


python ./db_utils/translate_db.py \
    --src_lang="en" \
    --tgt_lang="kr" \
    --src_db_path="./db_en" \
    --tgt_db_path="./db_kr" \
    --slot_alignment_path="./slot_alignment.json" \
    --value_alignment_path="./db_utils/outputs/kr2en_alignment.json"

python ./db_utils/translate_db_results.py \
    --src_lang="en" \
    --tgt_lang="kr" \
    --src_data_path="./dataset_json" \
    --tgt_data_path="./dataset_json" \
    --src_db_path="./db_en" \
    --tgt_db_path="./db_kr" \
    --output_path='./db_results' \
    --value_alignment_path="./db_utils/outputs/kr2en_alignment.json" \
    --debug
