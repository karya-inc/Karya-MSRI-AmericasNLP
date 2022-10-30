# Karya-MSRI@AmericasNLP

## Environment Specifications 

Both the translate and asr inference/training require separate environments that can be installed using the following requirement specifications. 

### For Translate 
```
pip3 install -r translate_requirements.txt
```

### For ASR
For Translate: 
```
pip3 install -r asr_requirements.txt
```

## Scripts 

```
├── src
│   ├── asr.py                          # Fine-tuning ASR models & utilizing KenLM during training 
│   ├── construct_kenlm.py              # Constructing kenlm data + training kenlm models 
│   ├── debug                           # WIP
│   │   └── asr_with_kenlm.py
│   ├── inference.py                    # Inference for ASR Models (KenLM support to be added) 
│   ├── translate.py                    # Training for the translate models 
│   ├── translate_inference.py          # Inference for the translation models 
│   └── translate_preprocess.py         # Preprocessing script to convert free-form data to HuggingFace Dataset form

├── scripts
│   ├── automate.sh                     #Automating the training of the asr models 
│   ├── inference.sh                    #Automating the inference of the asr models
│   ├── train_kenlm.sh                  #Automating the training of the kenlm models
│   ├── translate.sh                    #Automating the training of the translate models
│   └── translate_inference.sh          #Automating the inference of the asr models

```


Baseline models and loss curves can be found [here](https://drive.google.com/drive/folders/1I9s1kGzggu-UKvOjmE_kdK6GYl1Iy6Qi?usp=sharing)

Language Codes:

- Bribri - bzd
- Guarani - gn
- Kotiria - gvc
- Quechua - quy
- Wa'ikhana - tav
