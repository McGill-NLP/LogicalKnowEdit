#!/usr/bin/env python

from dataclasses import replace
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right
from typing import Callable, Dict, Iterable, List, Tuple, Union
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
import re

def prepare_bart_model(config):
    model = AutoModelForSeq2SeqLM.from_pretrained(config.eval.base_lm_path).cuda()
    #model.config.forced_bos_token_id = None
    if config.eval.bart_qa_path:
        state_dict = torch.load(config.eval.bart_qa_path, map_location="cpu")['state_dict']
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            state_dict = {re.sub("^model.", "", k): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
    model.train(False)
    tokenizer = AutoTokenizer.from_pretrained(config.eval.base_lm_path)
    if 'mend' in [config.eval.estab_method, config.eval.update_method]:
        gen_args = {
                "num_beams": 1,
                #"eos_token_id": (tokenizer.eos_token_id),
                #"min_length": 1,
                "max_length": 12,
            }
    else:
        gen_args = {
                "num_beams": 5,
                #"eos_token_id": (tokenizer.eos_token_id),
                #"min_length": 1,
                "max_length": 12,  
        }
    return model, tokenizer, gen_args

def prepare_gpt_model(config):
    model = AutoModelForCausalLM.from_pretrained(config.eval.base_lm_path).cuda()
    model.train(False)
    tokenizer = AutoTokenizer.from_pretrained(config.eval.base_lm_path)
    gen_args = {
        "num_beams": 1,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "max_length": 60,
    }
    
    return model, tokenizer, gen_args

def prepare_another_model(config):
    raise NotImplementedError

prepare_lm_func = {
    'bart': prepare_bart_model,
    'gpt': prepare_gpt_model,
    # add your new base lm here
}


def make_rule_data(rule: Dict, rule_format=None):
    assert rule_format in ["pre1", "pre2", "imp"]
    if mode == "pre1":
        src_data = ["If <mask>, and {0}, then {1}".format(rule["pre2"], rule["imp"])]
        tgt_data = [rule["pre1"]]
    
    elif mode == "pre2":
        src_data = ["If {0}, and <mask>, then {1}".format(rule["pre1"], rule["imp"])]
        tgt_data = [rule["pre2"]]

    elif mode == "imp":
        src_data = ["If {0}, and <mask>, then {1}".format(rule["pre1"], rule["imp"])]
        tgt_data = [rule["imp"]]

    return src_data, tgt_data

def get_lm_belief(query, tokenizer, base_lm, gen_args, lm_type):
    prefix = getattr(base_lm.config, "prefix", "") or ""
    query = [prefix + x for x in query if not x is None]
    if len(query) == 0:
        return []
    if lm_type == 'gpt':
        tokenizer.padding_side = "left"

    src_batch = tokenizer(query, return_tensors="pt", truncation=True, padding="longest").to('cuda')
    predictions = base_lm.generate(
        input_ids=src_batch.input_ids,
        attention_mask=src_batch.attention_mask,
        min_length=(1 if lm_type=='bart' else src_batch.input_ids.size(1)+1),
        **gen_args
    )
    
    if lm_type == 'gpt':
        predictions = tokenizer.batch_decode(predictions[:, src_batch.input_ids.size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(predictions[:2])
        tokenizer.padding_side = "right"
    else:
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return predictions
    
    