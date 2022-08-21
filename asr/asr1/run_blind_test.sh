#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=gn # bzd (Bribri), gn (Guarani), gvc (Kotiria), quy (Quechua), tav (Wa'ikhana)

train_set="train_${lang}"
valid_set="dev_${lang}"
test_sets="dev_${lang}"

asr_config=conf/train_asr_transformer.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang ${lang} \
    --ngpu 1 \
    --nbpe 100 \
    --stage 3 \
    --stop_stage 12 \
    --use_lm false \
    --max_wav_duration 38 \
    --speed_perturb_factors "1.0" \
    --skip_train true \
    --feats_normalize utterance_mvn \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model valid.cer_ctc.best.pth \
    --inference_nj 1 \
    --gpu_inference true \
    --local_data_opts "--lang ${lang}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
