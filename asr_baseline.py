import datasets 
from datasets import load_dataset, load_metric 
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
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, TrainingArguments, Trainer, Wav2Vec2ForCTC
from datasets import ClassLabel
import random
import pandas as pd
import torch
import csv 
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import argparse
cers = [1.0]

wandb.init(project="AmericasNLP-Baselines", entity="hdiddee")

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
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)       
    predictions = [pred.strip() for pred in pred_str]
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    output_prediction_file = os.path.join(f'/home/t-hdiddee/asr/models_src_raw/{args.lang}/', f'{wer}--{cer}_generated_predictions.txt')
    ## Saving the best CER Model 
    if cer < min(cers): 
        print('Replacing existing best model w.r.t to CER')
        trainer.save_model(f'../asr/models_src_raw/{args.lang}/best_cer_model')
    
    cers.append(cer)
    print(cer)
    print(min(cers))

    with open(output_prediction_file, "w+", encoding="utf-8") as writer:
        writer.write("\n".join(predictions))

    return {"wer": wer, "cer": cer}

def generate_vocab(train_dataset, eval_dataset, lang):
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
    
    with open(f'/home/t-hdiddee/asr/models_src_raw/{lang}/vocab.json', 'w+') as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False)

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type = str)
    parser.add_argument("--save_path", type = str)
    chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
      
    args = parser.parse_args()
    train_file_path = make_custom_hf_dataset(meta_root = f'/home/t-hdiddee/data/americasnlp/{args.lang}/train/', train_flag = True, save_path = args.save_path, lang = args.lang)
    dev_file_path = make_custom_hf_dataset(meta_root = f'/home/t-hdiddee/data/americasnlp/{args.lang}/dev/', train_flag = False,  save_path = args.save_path, lang = args.lang)
    
    # Loading into HF datasets - from our own CSV 
    data_files = {"train": train_file_path, "dev" : dev_file_path}
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]

    generate_vocab(train_dataset=train_dataset, eval_dataset = eval_dataset, lang = args.lang)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(f'/home/t-hdiddee/asr/models_src_raw/{args.lang}/', unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    train = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    test = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names)

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
    output_dir=f'../asr/models_src_raw/{args.lang}',
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
        eval_dataset =test,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

