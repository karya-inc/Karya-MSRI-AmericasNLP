from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from transformers import AutoProcessor
import argparse 
import tqdm 
import re
import torch
import command 
import pandas as pd
import numpy as np
from pyctcdecode import build_ctcdecoder
from datasets import load_dataset
import torchaudio
import os 

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

def make_test_hf_dataset(test_audio_path, lang, save_path):
    data = []
    with open(test_audio_path + 'meta.tsv') as file: 
        files_set = file.read().split('\n') 
        for wav_name in files_set: 
            try: 
                wav_path = test_audio_path + wav_name 
                # This is simply to check if the file being loaded is a valid audio file, only then load the transcription
                sample = torchaudio.load(wav_path)
                data.append({"wav_path": wav_path})
            except: 
                print(f'Could not read {wav_name}')

    df = pd.DataFrame(data)    
    file_path = f"{save_path}test_{lang}.tsv"
    df.to_csv(file_path, sep="\t", encoding="utf-8", index=False)
    
    return file_path

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

def prepare_dataset(batch):
    audio = torchaudio.load(batch['wav_path'])
    batch["input_values"] = processor(audio[0], sampling_rate=16_000).input_values[0][0]
    batch["input_length"] = len(batch["input_values"])
    return batch

def make_custom_hf_dataset(audio_path, lang, save_path):
    data = []
    with open(audio_path + 'meta.tsv') as file: 
        files_set = file.read().split('\n') 
        transcriptions = []
        for ele in files_set: 
            try:
                wav_name, src_processed, source_raw, target_raw = ele.split('\t')
                transcriptions.append(source_raw)
                wav_path = audio_path + wav_name 
                # This is simply to check if the file being loaded is a valid audio file, only then load the transcription
                sample = torchaudio.load(wav_path)
                data.append({"wav_path": wav_path})
            except: 
                print(f'Could not read {wav_name}')

    df = pd.DataFrame(data)    
    file_path = f"{save_path}test_{lang}.tsv"
    df.to_csv(file_path, sep="\t", encoding="utf-8", index=False)
    
    data_files = {"test": file_path}
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    
    test_dataset = dataset["test"]
    test = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names)
    return test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type = str)
    parser.add_argument("--transcriptions_path", type = str)
    parser.add_argument("--test_audio_path", type = str)
    parser.add_argument("--audio_path", type = str)
    parser.add_argument("--lang", type = str)
    parser.add_argument("--save_path", type = str)
    parser.add_argument("--inference_path", type = str)

    args = parser.parse_args()

    kenlm_train_file_path = prepare_kenlm_data(args.transcriptions_path, args.lang)
    kenlm_model_path = f'{args.lang}_5gram.arpa'
    os.system(f"./kenlm/build/bin/lmplz -o 3 <{kenlm_train_file_path} > {kenlm_model_path}")
    corrected_kenlm_model_path = clean_kenlm_model(kenlm_model_path)

    model = Wav2Vec2ForCTC.from_pretrained(args.model_path)
    processor = Wav2Vec2Processor.from_pretrained(args.model_path, pad_token = None) #This will be used to extract our processors vocab and feature extractor 
    test_data = make_custom_hf_dataset(args.audio_path, args.lang, args.save_path)
    
    vocab_dict = processor.tokenizer.get_vocab()
    print(vocab_dict)
    print(len(vocab_dict))
    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path=corrected_kenlm_model_path,
    )

    processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder)


   
    predictions = []
    for sample in tqdm.tqdm(test_data):
        input_dict = processor_with_lm(sample["input_values"], return_tensors="pt", sampling_rate=16_000)

        logits = model(input_dict.input_values).logits
        print(logits)
        print(logits.size())
        pred_str = processor_with_lm.decode(logits[0])  
        # print(pred_str)
        # pred_str = pred_str.replace('[PAD]','')
        # pred_str = pred_str.replace('[UNK]','')
        # predictions.append(pred_str)

# with open(args.inference_path, 'w') as file: 
#     for prediction in predictions: 
#         file.write(prediction + '\n')