import argparse 
import tqdm 
import torch
import command 
import pandas as pd
import numpy as np
from pyctcdecode import build_ctcdecoder
import os 

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

def clean_kenlm_model(kenlm_model_path):
    corrected_kenlm_model_path = f"corrected_{kenlm_model_path}"
    with open(kenlm_model_path, "r") as read_file, open(corrected_kenlm_model_path, "w") as write_file:
        has_added_eos = False
        for line in read_file:
            if not has_added_eos and "ngram 1=" in line:
                count=line.strip().split("=")[-1]
                write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
            elif not has_added_eos and "<s>" in line:
                write_file.write(line)
                write_file.write(line.replace("<s>", "</s>"))
                has_added_eos = True
            else:
                write_file.write(line)
    return corrected_kenlm_model_path

def prepare_kenlm_data(transcriptions_path, lang):
    data = [] 
    kenlm_train_file_path = f'{lang}_kenlm_train.txt'
    with open(transcriptions_path) as file: 
        files_set = file.read().split('\n') 
        transcriptions = []
        for ele in files_set: 
            try:
                wav_name, src_processed, source_raw, target_raw = ele.split('\t')
                source_raw = re.sub(chars_to_remove_regex, '', source_raw).lower()
                transcriptions.append(source_raw)
            except: 
                print(f'Failed for {ele}')
        print(f'{len(transcriptions)} number of transcriptions considered for creating the n-gram.')            
        with open(kenlm_train_file_path, "w") as file:
            file.write(" ".join(transcriptions))
        print('Training File for KenLM Written!')
    return kenlm_train_file_path
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--transcriptions_path", type = str)
    parser.add_argument("--lang", type = str)
    parser.add_argument("--kenlm_model_path", type = str)

    args = parser.parse_args()

    kenlm_train_file_path = prepare_kenlm_data(args.transcriptions_path, args.lang)
    kenlm_model_path = f'{args.kenlm_model_path}/{args.lang}_5gram.arpa'
    os.system(f"./kenlm/build/bin/lmplz -o 3 <{kenlm_train_file_path} > {kenlm_model_path}")
    corrected_kenlm_model_path = clean_kenlm_model(kenlm_model_path)
    print(f'Corrected KenLM saved at {corrected_kenlm_model_path}!')
    

 