"""
Rewriter Inference
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3950"

import torch
import sys
import time
from tqdm import tqdm
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
import transformers

transformers.logging.set_verbosity_error()
import json
from copy import deepcopy

# custom
import argparse, logging
from torch.utils.data import Dataset, DataLoader
import panda as pd

class PromptDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

###################################################

def parse_args():
    """
    Parses the arguments
    """
    
    parser = argparse.ArgumentParser(description="Run Inference Model.")
    
    parser.add_argument('--prompt_path', nargs='?', default='',
                        help='Specify the prompts file')
    
    return parser.parse_args()

def inference_LLM_patch(prompt_path):

    pretrained_path = 'xxx/Qwen2.5-72B-Instruct'

    model = AutoModelForCausalLM.from_pretrained(pretrained_path, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    model.half()

    data = []
    test_basic_info = pd.read_json(prompt_path)
    data = test_basic_info["extractor_prompt"].tolist() + test_basic_info["inferer_prompt"].tolist()     # get prompt
    instance_num = len(test_basic_info["extractor_prompt"].tolist())

    dataset = PromptDataset(data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    response_list = []
    for batch in tqdm(dataloader):
        try:
            text_batch = []
            for prompt in batch:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                text_batch.append(text)
            
            model_inputs = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            response_list.extend(response)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Out of Memory. {e}")
            response_list.extend(["error"] * len(batch))
            torch.cuda.empty_cache()
    
    #### save output ####
    test_basic_info["rewriter_extractor"] = response_list[:instance_num]
    test_basic_info["rewriter_inferer"] = response_list[instance_num:]

    test_basic_info.to_json("test_rewriter_output.json", index=False)

    return True

if __name__ == "__main__":

    args = parse_args()
    
    print("Start Rewiter Running:")
    inference_LLM_patch(args.prompt_path)
    print("Rewiter Running Ended.")