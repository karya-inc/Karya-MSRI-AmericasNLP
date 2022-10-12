LANGS = ('Kotiria')
for lang in "${LANGS}"
do
    echo "Training for $lang"
    kenlm/build/bin/lmplz -o 5 < "${lang}_kenlm_train.txt" > "${lang}_5gram.arpa"
done
