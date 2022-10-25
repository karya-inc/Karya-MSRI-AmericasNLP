#!/bin/bash
python ../asr.py --lang Bribri --save_path Bribri/ --kenlm_model_path ~/Karya-MSRI-AmericasNLP/lang_kenlm/Bribri_2gram.arpa_corrected.arpa 
python ../asr.py --lang Guarani --save_path Guarani/ --kenlm_model_path ~/Karya-MSRI-AmericasNLP/lang_kenlm/Guarani_2gram.arpa_corrected.arpa 
python ../asr.py --lang Quechua --save_path Quechua/ --kenlm_model_path ~/Karya-MSRI-AmericasNLP/lang_kenlm/Quechua_2gram.arpa_corrected.arpa
python ../asr.py --lang Waikhana --save_path Waikhana/ --kenlm_model_path ~/Karya-MSRI-AmericasNLP/lang_kenlm/Waikhana_2gram.arpa_corrected.arpa
python ../asr.py --lang Kotiria --save_path Kotiria/ --kenlm_model_path ~/Karya-MSRI-AmericasNLP/lang_kenlm/Kotiria_2gram.arpa_corrected.arpa
