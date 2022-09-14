## Feature Fusion 

- Has character based RNNLM

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnnlm_char_lm_lm_train_lm_rnn_gvc_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_gvc|253|2206|25.7|68.0|6.3|16.6|90.8|100.0|
|decode_asr_rnnlm_char_lm_lm_train_lm_rnn_bzd_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_bzd|250|2056|34.6|57.9|7.5|16.8|82.2|99.2|
|decode_asr_rnnlm_char_lm_lm_train_lm_rnn_gn_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_gn|93|391|1.3|74.2|24.6|10.2|109.0|100.0|
|decode_asr_rnnlm_char_lm_lm_train_lm_rnn_quy_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_quy|250|11465|45.8|50.4|3.8|11.2|65.4|100.0|
|decode_asr_rnnlm_char_lm_lm_train_lm_rnn_tav_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_tav|250|1201|8.4|86.5|5.1|27.1|118.7|99.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnnlm_char_lm_lm_train_lm_rnn_gvc_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_gvc|253|13453|71.0|15.3|13.7|12.5|41.5|100.0|
|decode_asr_rnnlm_char_lm_lm_train_lm_rnn_bzd_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_bzd|250|10083|70.8|16.2|13.0|10.3|39.5|99.2|
|decode_asr_rnnlm_char_lm_lm_train_lm_rnn_gn_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_gn|93|2946|28.9|33.6|37.6|7.6|78.8|100.0|
|decode_asr_rnnlm_char_lm_lm_train_lm_rnn_quy_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_quy|250|95334|84.5|9.1|6.4|10.4|25.9|100.0|
|decode_asr_rnnlm_char_lm_lm_train_lm_rnn_tav_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_tav|250|8606|61.8|21.5|16.7|15.1|53.2|99.6|