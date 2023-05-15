for split in test valid train fewshot ; do
    python dialogues/risawoz/src/convert.py --src en --tgt kr --setting kr --splits ${split} --debug
done


for split in test valid train fewshot ; do
    python dialogues/risawoz/src/preprocess.py --max_history 2 --last_two_agent_turns --gen_full_state --only_user_rg --sampling balanced --setting kr --fewshot_percent 0 --version 1 --splits ${split}
done

for split in test valid train fewshot ; do
    python3 dialogues/risawoz/scripts/check_entity.py --directory dialogues/risawoz/data/preprocessed/ --setting kr --version 1 --splits ${split}
done
