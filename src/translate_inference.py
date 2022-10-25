from transformers import ( PreTrainedTokenizerFast, MarianMTModel, MarianConfig, MT5ForConditionalGeneration, 
T5Tokenizer,MBartForConditionalGeneration, MBart50TokenizerFast)
import argparse
import tensorflow as tf
import tqdm
import torch
import time 
import numpy as np
import io
import os

predictions = []

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def seq_online(model, tokenizer, src_samples, task_prefix, return_tensor):
    samples = [task_prefix + sample for sample in src_samples]
    batch = tokenizer(samples, return_tensors=return_tensor, truncation=True, padding='max_length', max_length = 48) 
    output = model.generate(**batch, max_new_tokens = 48)

    predictions = tokenizer.batch_decode(output, skip_special_tokens=True)
    return predictions 


   
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str, default = None)
    parser.add_argument("--tgt_lang", type=str, default = None)
    parser.add_argument("--benchmark_path", type=str, default = './inference.txt')
    parser.add_argument("--return_tensor", type=str, default = 'pt')
    parser.add_argument("--model_arch", type=str, default = 'mt5')
    parser.add_argument("--model_path", type=str, default = None)
    parser.add_argument("--vocab_path", type=str, default = None)
    parser.add_argument("--task_prefix", type = str, default = "")
    parser.add_argument("--src_file", type=str)

    args = parser.parse_args()
    if "mt5" in args.model_arch: 
        tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        model = MT5ForConditionalGeneration.from_pretrained(args.model_path)
        assert len(args.task_prefix) > 2, "Haven't passed a task prefix for mt5-type model. Please pass task prefix."
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.vocab_path, bos_token = "<s>", eos_token = "</s>", pad_token = "<pad>", unk_token = "<unk>")
        model =  MarianMTModel.from_pretrained(pretrained_model_name_or_path = args.model_path)    
    
    src_samples = io.open(args.src_file, encoding='UTF-8').read().strip().split('\n')
    predictions = seq_online(model, tokenizer, src_samples, args.task_prefix, args.return_tensor)
    with open(args.benchmark_path, 'w+', encoding='UTF-8' ) as file:
        for pred in predictions:
            file.write(pred)
            file.write('\n')