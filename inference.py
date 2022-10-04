from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
import argparse 
import tqdm 
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
import torchaudio
# from asr_baseline import DataCollatorCTCWithPadding, prepare_dataset, make_custom_hf_dataset

def make_custom_hf_dataset(test_audio_path, lang, save_path):
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


def prepare_dataset(batch):
        audio = torchaudio.load(batch['wav_path'])
        batch["input_values"] = processor(audio[0], sampling_rate=16_000).input_values[0][0]
        batch["input_length"] = len(batch["input_values"])
        return batch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type = str)
    parser.add_argument("--processor_path", type = str)
    parser.add_argument("--test_audio_path", type = str)
    parser.add_argument("--lang", type = str)
    parser.add_argument("--save_path", type = str)
    parser.add_argument("--inference_path", type = str)

    args = parser.parse_args()

    model = Wav2Vec2ForCTC.from_pretrained(args.model_path).to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(args.model_path)
    
    test_file_path = make_custom_hf_dataset(args.test_audio_path, args.lang, args.save_path)
    data_files = {"test": test_file_path}
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    
    test_dataset = dataset["test"]
    test = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names)

    predictions = []

    for sample in tqdm.tqdm(test):    
        input_dict = processor(sample["input_values"], return_tensors="pt", padding=True, sampling_rate=16_000)

        logits = model(input_dict.input_values.to("cuda")).logits
        pred_ids = torch.argmax(logits, axis=-1)[0]
        pred_str = processor.decode(pred_ids, skip_special_tokens=True)  
        pred_str = pred_str.replace('[PAD]','')
        pred_str = pred_str.replace('[UNK]','')
        predictions.append(pred_str)

with open(args.inference_path, 'w') as file: 
    for prediction in predictions: 
        file.write(prediction + '\n')