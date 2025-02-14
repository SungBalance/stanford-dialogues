# Dialogues
This codebase provides a unified interface to several dialogue datasets.

# Available datasets:
- BiToD
- RiSAWOZ


# Data
Adding a new language:
- Look at `risawoz/data/original/en_{split}.json` to understand how the dataset is formatted.
- "user_utterance" and "system_utterance" contain the following: 1) utterance in natural language 2) entities and their word-spans in the utterance
- "db_results" contain the retrieved entries from database when agent makes an api call
- For a new language, the following needs to be done:
- Data Translation:
  - "user_utterance" and "system_utterance" should be translated to the target language. We let translators use their best judgment on how each entity should be translated given the context.
  - Translators need to annotate span of entities in the translated sentences which would be used to create a mapping between source entities and target entities. We have a UI tool that aids translators in doing so.
  - This results in a one-to-many mapping as each source entity may have multiple translations. For reasons mentioned later, we need to choose one of the translations as the canonical one and create a second mapping to map all possible translations to the canonical one (en2canonical.json).
- Database Translation:
  - The entity values in English database needs to be translated to the target langauge according to the alignment information. You can use the second mapping to do this.
  - Similarly, you can use this mapping to translate "db_results" in the dataset.

# Validating your work:
- Once you've created the dataset in the target language, put the content in the following file `risawoz/data/original/{language}_{split}.json`
- Add the new database files under `risawoz/database/db_{lang}/`. Follow the same formatting as English. The slot names don't need to be translated, only slot values.
- Run `python3 dialogues/risawoz/src/convert.py --setting {language} --splits {split}` to convert your data into a format suitable for preprocessing.
- You'll likely see the following in your output logs "API call likely failed for...". This could mean many things ranging from wrong alignment during translation, mismatch between entities in the translated sentence and database values, etc. To have a more accurate check, we restore the API calls from the existing search results. You should also try your best to solve the failed API calls by correcting belief states annotations and the two mappings. Our script will show some clues for you to solve these issues (e.g., the mismatches between the belief states and ground-truth search results which make an API call fail). If you get only a few of these errors, it means that the translated dataset is already of relatively high quality.
- If conversions is successful, you will see the converted file: `risawoz/data/{language}_{split}.json`. You can check the file to make sure it looks good.
- Run `python3 dialogues/risawoz/src/preprocess.py --max_history 2 --last_two_agent_turns --gen_full_state --only_user_rg --sampling balanced --setting {lang} --fewshot_percent 0 --version 1 --splits {split}` to preprocess the data for training.
- If preprocessing is successful, you will see the resulting file: `risawoz/data/preprocessed/{language}_{split}.json`.
- Run `python3 dialogues/risawoz/scripts/check_entity.py --directory dialogues/risawoz/data/preprocessed/ --version 1 --splits {split}` to sanity check the data. This script ensures that entities in the output are present in the input. This is necessary since our models are trained to copy entities from the input. This script will create a file `dialogues/risawoz/data/preprocessed/{split}_entity_check_1.tsv` including erroneous turns.
- To fix erroneous turns, you need to backtrack and sanity check every step of the data processing until you find the bug.
- If everything passes without errors, congrats! We will soon have a dialogue agent that can speak your language!
