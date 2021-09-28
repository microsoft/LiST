path=/scratch/gilbreth/wang5075/Project/PromptST/data/clue
for task in MNLI RTE
do
for SEED in 1 2 3 4 5
do
for SHOT in 500 1000
do
cat ${path}/${task}/${SHOT}-${SEED}/train.tsv <(tail -n +2 ${path}/${task}/${SHOT}-${SEED}/dev.tsv) > clue/${task}/${SHOT}-${SEED}/train.tsv
done
done
done