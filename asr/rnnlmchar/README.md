## Baseline + Character tokeniser +  RNNLM

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_rnn_bzd_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_bzd|250|2056|29.5|60.7|9.8|12.9|83.4|100.0|
|decode_asr_lm_lm_train_lm_rnn_gn_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_gn|93|391|26.6|62.7|10.7|6.9|80.3|97.8|
|decode_asr_lm_lm_train_lm_rnn_quy_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_quy|250|11465|45.5|49.7|4.8|8.6|63.2|100.0|
|decode_asr_lm_lm_train_lm_rnn_tav_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_tav|250|1201|6.2|88.3|5.4|21.8|115.6|99.6|


### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_rnn_bzd_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_bzd|250|10083|63.5|17.8|18.7|6.4|42.9|100.0|
|decode_asr_lm_lm_train_lm_rnn_gn_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_gn|93|2946|85.1|8.2|6.7|4.0|18.8|97.8|
|decode_asr_lm_lm_train_lm_rnn_quy_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_quy|250|95334|83.7|8.8|7.5|8.6|24.9|100.0|
|decode_asr_lm_lm_train_lm_rnn_tav_char_valid.loss.ave_asr_model_valid.cer_ctc.best/dev_tav|250|8606|58.1|21.7|20.2|12.6|54.5|99.6|

<!-- ### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err| -->