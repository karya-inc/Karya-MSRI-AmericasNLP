# Karya-MSRI@AmericasNLP

## Environment Specifications 

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


├── scripts
│   ├── automate.sh                     #Automating the training of the asr models 
│   ├── inference.sh                    #Automating the inference of the asr models
│   ├── train_kenlm.sh                  #Automating the training of the kenlm models

```
## Detailed Hyperparameter Setups 
Training Curves (and hyp setups) for the best performing models on the dev set can be found [here](https://wandb.ai/hdiddee/AmericasNLP-KENLM?workspace=user-hdiddee)

## Baseline models and loss curves can be found [here](https://drive.google.com/drive/folders/1I9s1kGzggu-UKvOjmE_kdK6GYl1Iy6Qi?usp=sharing). 


## Models for use with the HuggingFace API can be found [here](https://huggingface.co/HarshitaDiddee).

Language Codes:

- Bribri - bzd
- Guarani - gn
- Kotiria - gvc
- Quechua - quy
- Wa'ikhana - tav
