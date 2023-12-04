#!/usr/bin/env python

import argparse
from dataclasses import replace
import datetime
import json
import time

import warnings
from logging import getLogger
import sys
from pathlib import Path
import os
#from typing import Callable, Dict, Iterable, List, Tuple, Union
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
import torch
from torch import optim
from torch import nn
from tqdm import tqdm
import numpy as np
import copy
import re
from collections import defaultdict
import random
import hydra
from omegaconf import OmegaConf
from eval_utils import prepare_lm_func, get_lm_belief #eval_lm, 
from editing import prepare_edit_method
import utils

OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())

logger = getLogger(__name__)


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Metric():
    hit_org_fact = []
    hit_org_infer_ent = []
    hit_org_infer_sent = []
    hit_rephrase_fact = []
    hit_rephrase_infer_ent = []
    hit_rephrase_infer_sent = []


def sample_random_facts(data_samples, cur_data, n=5):
    data_copy = copy.deepcopy(data_samples)
    random.shuffle(data_copy)

    cur_relations = set([x['trips'][1] for x in cur_data['init']['facts']])
    for ex_ in data_copy:
        this_relations = set([x['trips'][1] for x in ex_['init']['facts']])
        if len(cur_relations.intersection(this_relations)) == 0:
            return random.sample(ex_["init"]["facts"], n)

def write_lm_memo(data, memo):
    def _ins_dict_(dict_, k, v):
        dict_[k] = v
        return dict_
    return [_ins_dict_(x, 'memo', y) for (x, y) in zip(data, memo)] # _example_split['memo'] = y

def write_new_imp_memo(_example_split_memo, _example_init):
    def _ins_dict_(_dict, ture_or_false):
        _dict["new_imp"] = ture_or_false
        return _dict
    return [_ins_dict_(x, True) if x["q"]!=y["q"] else _ins_dict_(x, False) for (x, y) in zip(_example_split_memo, _example_init)] # _example_split['memo'] = y


def memo_lm(config, data, tokenizer, base_lm, gen_args):
    data_memo = copy.deepcopy(data)
    # TODO: create an record object
    fact_memo = get_lm_belief(
        [x["q"] for x in sorted(
            data_memo["init"]["queries"]['original']["facts"],
            key=lambda d: d['q']
            )
        ],
        tokenizer, 
        base_lm, 
        gen_args, 
        config.eval.lm_type
    )

    tmp_a = [x["q"] for x in sorted(
            data_memo["init"]["queries"]['semantic-equiv']["facts"],
            key=lambda d: d['q']
            )
        ]
    fact_rephrase_memo = get_lm_belief(
        tmp_a, 
        tokenizer, 
        base_lm, 
        gen_args, 
        config.eval.lm_type
    )

    infer_memo = get_lm_belief(
        [x["q"] for x in sorted(
            data_memo["init"]["queries"]['original']["inference"],
            key=lambda d: d['q']
            )
        ],
        tokenizer, 
        base_lm, 
        gen_args, 
        config.eval.lm_type
    )

    infer_rephrase_memo = get_lm_belief(
        [x["q"] for x in sorted(
            data_memo["init"]["queries"]['semantic-equiv']["inference"],
            key=lambda d: d['q']
            )
        ],
        tokenizer, 
        base_lm, 
        gen_args, 
        config.eval.lm_type
    )

    for t in list(data_memo.keys()):
        #print("split", t)
        if t in ["score", "gp_id"]:
            continue
        data_memo[t]['queries']["original"]['facts'] = write_lm_memo(
                sorted(
                    data_memo[t]['queries']["original"]['facts'],
                    key=lambda d: d['q']
                ), 
                fact_memo
        )
        tmp_b = sorted(
                    data_memo[t]['queries']["semantic-equiv"]['facts'],
                    key=lambda d: d['q']
                )

        assert tmp_a == [x["q"] for x in tmp_b]
        data_memo[t]['queries']["semantic-equiv"]['facts'] = write_lm_memo(
                tmp_b, 
                fact_rephrase_memo
        )

        data_memo[t]['queries']["original"]['inference'] = write_lm_memo(
                    sorted(
                        data_memo[t]['queries']["original"]['inference'],
                        key=lambda d: d['q']
                    ),  
                    infer_memo
            )

        data_memo[t]['queries']["original"]['inference'] = write_new_imp_memo(
                data_memo[t]['queries']["original"]['inference'],
                data_memo["init"]['queries']["original"]['inference']
        )


        data_memo[t]['queries']["semantic-equiv"]['inference'] = write_lm_memo(
                sorted(
                    data_memo[t]['queries']["semantic-equiv"]['inference'],
                    key=lambda d: d['q']
                ),  
                infer_rephrase_memo
        )

    return data_memo


def memo_irre_lm(config, data, tokenizer, base_lm, gen_args):
    data_memo = copy.deepcopy(data)

    memo = get_lm_belief(
        [x["q"] for x in data_memo],
        tokenizer, 
        base_lm, 
        gen_args, 
        config.eval.lm_type
    )
    
    data_memo = write_lm_memo(data_memo, memo)
    return data_memo

def eval_memo_irre(config, data, tokenizer, base_lm, gen_args):
    predictions = get_lm_belief(
        [x['q'] for x in data if x["q"] is not None], 
        tokenizer, 
        base_lm, 
        gen_args, 
        config.eval.lm_type
    )
    memo_hit = [1. if _p==_t else 0. for i, (_p, _t) in enumerate(zip(
            predictions, 
            [x['memo'] for x in data]
            )
        )]
    return memo_hit

def gen_and_eval(eval_data, tokenizer, base_lm, gen_args, lm_type, verbose=False, for_imp=False):
    src_data = [x['q'] for x in eval_data if x["q"] is not None]
    tgt_data = [x['a'] for x in eval_data if x["q"] is not None]
    if len(src_data) == 0:
        return [], [], []
    
    predictions = get_lm_belief(src_data, tokenizer, base_lm, gen_args, lm_type)


    hit = [1. if _p==_t else 0. for (_p, _t) in zip(
        predictions, 
        [x['a'] for x in eval_data]
    )]
    update_hit = [_x for i, _x in enumerate(hit) if eval_data[i]['is_update']]
    keep_hit = [_x for i, _x in enumerate(hit) if not eval_data[i]['is_update']]
    
    if for_imp:
        memo_hit = [1. if _p==_t else 0. for i, (_p, _t) in enumerate(zip(
            predictions, 
            [x['memo'] for x in eval_data]
        )) if not eval_data[i]['is_update'] and not eval_data[i]["new_imp"]]
    else:
        memo_hit = [1. if _p==_t else 0. for i, (_p, _t) in enumerate(zip(
            predictions, 
            [x['memo'] for x in eval_data]
        )) if not eval_data[i]['is_update']]

    return update_hit, keep_hit, memo_hit

# gen_and_eval(eval_data, tokenizer, base_lm, gen_args, lm_type):
def eval_lm(data, tokenizer, base_lm, gen_args, lm_type, verbose=False):
    rslts = {}
    fact_data = data['queries']['original']['facts']
    rslts['hit_org_fact_update'], rslts['hit_org_fact_keep'], rslts['hit_org_fact_memo'] = gen_and_eval(
        fact_data, tokenizer, base_lm, gen_args, lm_type, verbose=False)

    #infer_data = [x for x in data['queries']['original']['inference'] if x["q"] is not None and x["trips"][0] != x["trips"][1]]
    #if config.eval.do_bc:
    #    tmp = proxy_eval_inference(
    #        base_lm,
    #        fact_data,
    #        data['queries']['original']['inference'],
    #        "original_inference"
    #    )
    #else:
    tmp = gen_and_eval(
        data['queries']['original']['inference'],
        tokenizer,
        base_lm,
        gen_args,
        lm_type,
        for_imp=True
    )
    rslts['hit_org_infer_ent_update'] = tmp[0]
    rslts['hit_org_infer_ent_keep'] = tmp[1]
    rslts['hit_org_infer_ent_memo'] = tmp[2]
    
    fact_data = data['queries']['semantic-equiv']['facts']
    rslts['hit_rephrase_fact_update'], rslts['hit_rephrase_fact_keep'], rslts['hit_rephrase_fact_memo'] = gen_and_eval(
                    data['queries']['semantic-equiv']['facts'], 
                    tokenizer,
                    base_lm,
                    gen_args,
                    lm_type
                    )

    tmp = gen_and_eval(
        data['queries']['semantic-equiv']['inference'], 
        tokenizer,
        base_lm,
        gen_args,
        lm_type,
        for_imp=True
    )
    rslts['hit_rephrase_infer_update'] = tmp[0]
    rslts['hit_rephrase_infer_keep'] = tmp[1]
    rslts['hit_rephrase_infer_memo'] = tmp[2]
    return rslts

@hydra.main(config_path='/home/lcc/knowedit_github/config', config_name='config_eval')
def main(config):
    print("Base MODEL Path:\t", config.eval.base_lm_path)

    torch.manual_seed(42)

    dataset = []
    with open(config.eval.input_path) as f:
        for line in f:
            dataset.append(json.loads(line))

    lm_type = config.eval.lm_type
    base_lm, tokenizer, gen_args = prepare_lm_func[lm_type](config)
    edit_methods = prepare_edit_method(config, base_lm, tokenizer)

    estab_rslts = defaultdict(lambda : [])
    update_rslts = defaultdict(lambda : [])

    for i, data in enumerate(tqdm(dataset)):
        # establish facts & rules
        estab_lm = edit_methods['estab'].edit_lm(data["init"]['facts'], rule_data=data["init"]["rule"])
        irre_memo = memo_irre_lm(
            config, 
            sample_random_facts(dataset, data, n=5), 
            tokenizer, 
            estab_lm, 
            gen_args)

        data_memo = memo_lm(config, data, tokenizer, estab_lm, gen_args)
        # eval_lm(data, tokenizer, base_lm, gen_args, lm_type, verbose=False)
        est_rslt = eval_lm(data_memo["init"], tokenizer, estab_lm, gen_args, lm_type, verbose=False)
        
        tmp = eval_memo_irre(config, irre_memo, tokenizer, estab_lm, gen_args)
        estab_rslts["irre_memo_hit"] = estab_rslts["irre_memo_hit"] + tmp
        for k, v in est_rslt.items(): estab_rslts[k] += v
        # further update
        for t in ['0', '1', '2']:
            edit_methods['update'].reload_lm(estab_lm)
            #if config.eval.do_fc:
            #    derived_facts = proxy_derivation(
            #        base_model, data_memo[t]['facts'], data_memo[t]['queries']['original']['inference']
            #    )
            #else:
            #    derived_facts = None
            update_lm = edit_methods['update'].edit_lm(
                [x for x in data_memo[t]['facts'] if x['is_update']]
                )

            upt_rslt = eval_lm(data_memo[t], tokenizer, update_lm, gen_args, lm_type, verbose=False)
            for k, v in upt_rslt.items(): 
                update_rslts[k] += v
            #update_rslts = upt_rslt if update_rslts is None else {k: v+_upt_rslts[k] for k, v in update_rslts.items()}
            update_rslts["irre_memo_hit"] += eval_memo_irre(config, irre_memo, tokenizer, update_lm, gen_args)
        
    print('base results:')
    for k, v in estab_rslts.items():
        print(k, np.mean(v))

    print('update results:')
    for k, v in update_rslts.items():
        print(k, np.mean(v))

if __name__ == "__main__":
    main()
