import argparse 
import tqdm 
import torch
import command 
import pandas as pd
import numpy as np
import re
from pyctcdecode import build_ctcdecoder
import os 

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

def clean_kenlm_model(kenlm_model_path):

    corrected_kenlm_model_path = f"{kenlm_model_path}_corrected.arpa"
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

def prepare_kenlm_data(train_path, dev_path, lang):
    data = [] 
    kenlm_train_file_path = f'{lang}_kenlm_train.txt'

    with open(train_path) as train_file: 
        train_files_set = train_file.read().split('\n')

        # print(f'Number of files for consideration: {len(files_set)}!')
        transcriptions = []
        for ele in train_files_set:
            try:
                wav_name, src_processed, source_raw, target_raw = ele.split('\t')
                source_raw = re.sub(chars_to_remove_regex, '', source_raw).lower()
                transcriptions.append(source_raw)
            except:        
                print(f'Failed for {ele}')
    
    with open(dev_path) as dev_file: 
        dev_files_set = dev_file.read().split('\n')

        for ele in dev_files_set:
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
    parser.add_argument("--train_path", type = str)
    parser.add_argument("--dev_path", type = str, default = None)
    parser.add_argument("--lang", type = str)
    parser.add_argument("--kenlm_model_path", type = str)

    args = parser.parse_args()

    kenlm_train_file_path = prepare_kenlm_data(args.train_path, args.dev_path, args.lang)
    kenlm_model_path = f'{args.kenlm_model_path}/{args.lang}_2gram.arpa'
    os.system(f"./kenlm/build/bin/lmplz -o 2 <{kenlm_train_file_path} > {kenlm_model_path} --discount_fallback")
    corrected_kenlm_model_path = clean_kenlm_model(kenlm_model_path)
    print(f'Corrected KenLM saved at {corrected_kenlm_model_path}!')
    

 