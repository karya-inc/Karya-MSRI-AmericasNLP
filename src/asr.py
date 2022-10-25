import datasets 
from datasets import load_dataset, load_metric 
import tqdm
from pyctcdecode import build_ctcdecoder
from dataclasses import dataclass, field
from typing import Optional
import transformers 
from evaluate import load 
import re
import json 
import numpy as np
import os 
import torchaudio
import wandb
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, TrainingArguments, Trainer, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from datasets import ClassLabel
import random
import pandas as pd
import torch
import csv 
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import argparse
cers = [1.0]
kenlm_cers = [1.0]

wandb.init(project="AmericasNLP-KENLM", entity="hdiddee")

## Helper Functions 
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

def remove_special_characters(batch):
    batch["transcript"] = re.sub(chars_to_remove_regex, '', batch["transcript"]).lower()
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["transcript"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def prepare_dataset(batch):
    audio = torchaudio.load(batch['wav_path'])
    batch["input_values"] = processor(audio[0], sampling_rate=16_000).input_values[0][0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids
    return batch 

        
def make_test_hf_dataset(meta_root, lang, save_path):
    data = []
    with open(meta_root + 'meta.tsv') as file: 
        files_set = file.read().split('\n') 
        for wav_name in files_set: 
            try: 
                wav_path = meta_root + wav_name 
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
    
    def prepare_test_dataset(batch):
        audio = torchaudio.load(batch['wav_path'])
        batch["input_values"] = processor(audio[0], sampling_rate=16_000).input_values[0][0]
        batch["input_length"] = len(batch["input_values"])
    
        return batch 

    test_dataset = dataset["test"]
    test = test_dataset.map(prepare_test_dataset, remove_columns=test_dataset.column_names)
    return test

def make_custom_hf_dataset(meta_root, train_flag, save_path, lang):
    data = []
    with open(meta_root + 'meta.tsv') as file: 
        files_set = file.read().split('\n') 
        for ele in files_set: 
            try: 
                wav_name, src_processed, source_raw, target_raw = ele.split('\t')
                wav_path = meta_root + wav_name   
                # This is simply to check if the file being loaded is a valid audio file, only then load the transcription
                sample = torchaudio.load(wav_path)
                data.append({"wav_path": wav_path, "transcript": source_raw})
            except: 
                print(f'Could not read {wav_name}')

    df = pd.DataFrame(data)    
    if train_flag: 
        file_path = f"{save_path}train_{lang}.tsv"
        df.to_csv(file_path, sep="\t", encoding="utf-8", index=False)
    else: 
        file_path = f"{save_path}dev_{lang}.tsv"
        df.to_csv(file_path, sep="\t", encoding="utf-8", index=False)
    
    return file_path
  

def compute_metrics(pred):
    pred_logits = pred.predictions

    # Without LM Outputs 
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)       
    predictions = [pred.strip() for pred in pred_str]
  
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    output_prediction_file = os.path.join(args.save_path, f'{cer}--{wer}_generated_predictions.txt')

    with open(output_prediction_file, "w+", encoding="utf-8") as writer:
        writer.write("\n".join(predictions))


    # With Kenlm Outputs 

    pred_str = processor_with_lm.batch_decode(pred_logits).text  
    predictions = [pred.strip() for pred in pred_str]
    kenlm_wer = wer_metric.compute(predictions=pred_str, references=label_str)
    kenlm_cer = cer_metric.compute(predictions=pred_str, references=label_str)
    print(f'{kenlm_cer, kenlm_wer} are the lm CER and WER respectively.')
    output_prediction_file = os.path.join(args.save_path, f'{kenlm_cer}--{kenlm_wer}_generated_predictions_with_kenlm.txt')

    with open(output_prediction_file, "w+", encoding="utf-8") as writer:
        writer.write("\n".join(predictions))

    ## Saving the best CER Model 
    if cer < min(cers): 
        print('Replacing existing best model w.r.t to no LM CER')
        trainer.save_model(f'{args.save_path}/best_cer_model')
    
    cers.append(cer)
    print(cer)


    if kenlm_cer < min(kenlm_cers): 
        print('Replacing existing best model w.r.t to kenLM CER')
        trainer.save_model(f'{args.save_path}/best_kenlm_cer_model')

    kenlm_cers.append(kenlm_cer)
    print(kenlm_cer)


    return {"wer": wer, "cer": cer, "kenlm_wer": kenlm_wer, "kenlm_cer": kenlm_cer}

def generate_vocab(train_dataset, eval_dataset, lang, save_path):
    train_dataset = train_dataset.map(remove_special_characters)
    eval_dataset = eval_dataset.map(remove_special_characters)

    vocab_train = train_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_dataset.column_names)
    vocab_test = eval_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=eval_dataset.column_names)

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print(len(vocab_dict))
  
    with open(f'{save_path}vocab.json', 'w+') as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False)

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type = str)
    parser.add_argument("--save_path", type = str)
    parser.add_argument("--kenlm_model_path", type = str)
    chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
    args = parser.parse_args()
      
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train_file_path = make_custom_hf_dataset(meta_root = f'/home/t-hdiddee/data/americasnlp/train_set/{args.lang}/train/', train_flag = True, save_path = args.save_path, lang = args.lang)
    dev_file_path = make_custom_hf_dataset(meta_root = f'/home/t-hdiddee/data/americasnlp/train_set/{args.lang}/dev/', train_flag = False,  save_path = args.save_path, lang = args.lang)
    
    # Loading into HF datasets - from our own CSV 
    data_files = {"train": train_file_path, "dev" : dev_file_path}
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]

    generate_vocab(train_dataset=train_dataset, eval_dataset = eval_dataset, lang = args.lang, save_path = args.save_path)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.save_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    
    test_data = make_test_hf_dataset(meta_root = f'/home/t-hdiddee/data/americasnlp/test_set/{args.lang}/test_inputs/', lang = args.lang , save_path = args.save_path)

   
    train = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    eval = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names)

    
    
    ## With Kenlm
    corrected_kenlm_model_path = args.kenlm_model_path
    vocab_dict = processor.tokenizer.get_vocab()
    print(vocab_dict)
    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path=corrected_kenlm_model_path,
    )

    processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    wer_metric = load("wer")
    cer_metric = load("cer")

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-xls-r-300m", 
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )


    model.freeze_feature_extractor()
    model.config.ctc_zero_infinity = True

    training_args = TrainingArguments(
    output_dir=args.save_path,
    overwrite_output_dir = True, 
    group_by_length=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=80,
    gradient_checkpointing=True,
    fp16=True,
    save_steps=100,
    eval_steps=50,
    logging_steps=50,
    learning_rate=3e-4,
    warmup_steps=300,
    save_total_limit=1,
    load_best_model_at_end = True, 
    skip_memory_metrics = True
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=eval,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()


    ## Test Set Evaluations
     

    # for sample in tqdm.tqdm(test_data):
    #     input_dict = processor_with_lm(sample["input_values"], return_tensors="pt", sampling_rate=16_000)
    #     logits = model(input_dict.input_values).logits
    #     print(logits.size)
    #     pred_str = processor_with_lm.batch_decode(logits)
    #     predictions.append(pred_str)
    #     print(pred_str)  

    # with open(f'{args.output_dir}/test_inference.txt', 'w') as file: 
    #     for pred in predictions: 
    #         file.write(pred)

    
