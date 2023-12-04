from transformers.models.bart.modeling_bart import shift_tokens_right
import copy
from mend.algs.mend import MEND
from memit import MEMITHyperParams, apply_memit_to_model
import torch
from torch import optim
from eval_utils import make_rule_data

class Editor:
    '''
    wrapper for your editing methods
    '''

    def __init__(self):
        super().__init__()

    def edit_lm(self):
        raise NotImplementedError

    def reload_lm(self):
        raise NotImplementedError

    def get_lm_belief(self):
        raise NotImplementedError

class NaiveEditor(Editor):
    def __init__(self, base_lm):
        self.base_lm = base_lm

    def edit_lm(self, *args, **kwargs):
        return self.base_lm

    def reload_lm(self, base_lm):
        self.base_lm = base_lm

class GPTFTEditor(Editor):
    def __init__(self, config, base_lm, tokenizer, loss_thld=1e-7):
        self.base_lm = base_lm
        self.ft_lr = config.eval.ft_lr
        self.use_rule = config.eval.use_rules
        self.max_ft_steps = config.eval.max_ft_step
        self.tokenizer = tokenizer
        self.loss_thld = loss_thld

    def calc_loss(self, base_lm, src_texts, tgt_texts):
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = self.tokenizer(
                [x + " " + y for x, y in zip(src_texts, tgt_texts)], 
                return_tensors="pt", 
                max_length=60,
                truncation=True, 
                padding="longest"
            )["input_ids"].cuda()

        outputs = base_lm(
                    input_ids, 
                    attention_mask=input_ids.ne(self.tokenizer.pad_token_id), 
                )
        lm_logits = outputs["logits"]
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
        loss = ce_loss_fct(lm_logits[:, :-1, :].contiguous().view(-1, lm_logits[:, :-1].shape[-1]), input_ids[:, 1:].contiguous().view(-1))

        return loss

    def edit_lm(self, fact_data, rule_data=None, **kwargs):
        prefix = getattr(self.base_lm.config, "prefix", "") or ""
        src_data = [prefix + x['q'] for x in fact_data]
        if len(src_data) == 0:
            return self.base_lm
        
        tgt_data = [x['a'] + self.tokenizer.eos_token for x in fact_data]
        if self.use_rule:
            rule_src_data, rule_tgt_data = make_rule_data(example["rule"], "imp")
            src_data += rule_src_data
            tgt_data += rule_tgt_data    
        
        new_lm = copy.deepcopy(self.base_lm)
        optimizer = optim.RMSprop(new_lm.parameters(), lr=self.ft_lr)
        for _ in range(self.max_ft_steps):
            optimizer.zero_grad()
            for _b in range(0, len(src_data), 2): # too big
                loss = self.calc_loss(new_lm, src_data[_b:_b+2], tgt_data[_b:_b+2])#[0]
                loss.backward()
            if loss < self.loss_thld:
                break
            optimizer.step()

        return new_lm

    def reload_lm(self, base_lm):
        self.base_lm = base_lm
 

class BartFTEditor(Editor):
    def __init__(self, config, base_lm, tokenizer, loss_thld=1e-7):
        self.base_lm = base_lm
        self.use_rule = config.eval.use_rules
        self.max_ft_steps = config.eval.max_ft_step
        self.tokenizer = tokenizer
        self.loss_thld = loss_thld
        self.ft_lr = config.eval.ft_lr

    def calc_loss(self, src_batch, tgt_batch, model):
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = src_batch["input_ids"], src_batch["attention_mask"]
        tgt_ids = tgt_batch["input_ids"]
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)

        outputs = model(
                src_ids, 
                attention_mask=src_mask, 
                decoder_input_ids=decoder_input_ids, 
                use_cache=False,
                output_hidden_states=False,
            )

        lm_logits = outputs["logits"]
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        return loss

    def edit_lm(self, fact_data, rule_data=None, **kwargs):
        
        #fact_data = [x for x in example['facts'] if x['is_update']] if do_update else example['facts']

        #if not derived_facts is None:
        #    fact_data += derived_facts
        prefix = getattr(self.base_lm.config, "prefix", "") or ""
        src_data = [prefix + x['q'] for x in fact_data] 
        #print('src_data', src_data)
        if len(src_data) == 0:
            return self.base_lm
        
        tgt_data = [x['a'] + self.tokenizer.eos_token for x in fact_data]

        if self.use_rule:
            rule_src_data, rule_tgt_data = make_rule_data(rule_data, "imp")
            src_data += rule_src_data
            tgt_data += rule_tgt_data  
        
        src_batch = self.tokenizer(src_data, return_tensors="pt", truncation=True, padding="longest").to('cuda')
        tgt_batch = self.tokenizer(tgt_data, return_tensors="pt", truncation=True, padding="longest").to('cuda')
        new_lm = copy.deepcopy(self.base_lm)
        optimizer = optim.RMSprop(new_lm.parameters(), lr=self.ft_lr)
        for _ in range(self.max_ft_steps):
            optimizer.zero_grad()
            loss = self.calc_loss(src_batch, tgt_batch, new_lm)#[0]
            if loss < self.loss_thld:
                break
            loss.backward()
            optimizer.step()
        return new_lm


    def reload_lm(self, base_lm):
        self.base_lm = base_lm

class MendEditor(Editor):

    def __init__(self, config, base_lm, tokenizer):
        self.mend_model = MEND(base_lm, config, lambda: copy.deepcopy(base_lm)).cuda()
        archive = torch.load(config.eval.mend_model_path, map_location="cpu")
        self.mend_model.load_state_dict(archive["model"])
        self.mend_model.train(False)
        self.tokenizer = tokenizer

    def edit_lm(self, fact_data, **kwargs):
        #fact_data = [x for x in kset['facts'] if x['is_update']] if do_update else kset['facts'] 
        prefix = getattr(self.mend_model.model.config, "prefix", "") or ""
        src_data = [prefix + x['q'] for x in fact_data]
        if len(src_data) == 0:
            return self.mend_model.model
        trg_data = [x['a'] for x in fact_data]
        src_batch = self.tokenizer(src_data, return_tensors="pt", truncation=True, padding="longest").to('cuda')
        trg_batch = self.tokenizer(trg_data, return_tensors="pt", truncation=True, padding="longest").to('cuda')

        mend_input = {
            "input_ids": src_batch["input_ids"],
            "attention_mask": src_batch["attention_mask"],
            "decoder_input_ids": trg_batch["input_ids"],
            "decoder_attention_mask": trg_batch["attention_mask"],
            "labels": trg_batch["input_ids"].masked_fill(trg_batch["input_ids"] == self.tokenizer.pad_token_id, -100) 
        }
        mend_input["decoder_input_ids"][:, 0] = self.tokenizer.eos_token_id
        edited_mend, _ = self.mend_model.edit(mend_input, detach_history=True)
        return edited_mend.model

    def reload_lm(self, base_lm):
        self.mend_model.model = base_lm


class MemitEditor:
    def __init__(self, config, base_lm, tokenizer):
        self.base_lm = base_lm
        self.tokenizer = tokenizer
        self.hparams = MEMITHyperParams.from_json(config.eval.memit_params_path)
        self.cache_template_dir = config.eval.memit_cache_template_dir

    def edit_lm(self, fact_data, **kwargs):
        cache_template = f"{self.cache_template_dir}/standup_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        #print(f"Will load cache from {cache_template}")
        args_conserve_memory = dict()
        etc_args = dict(cache_template=cache_template)
        requests = [{
                "case_id": i, 
                "prompt": x["q"].replace(x["trips"][0], "{}", 1),
                "subject": x["trips"][0],
                "target_new": {"str": x["a"].lstrip()+self.tokenizer.eos_token}
            } for i, x in enumerate(fact_data)
        ]
        
        requests = [x for x in requests if x["prompt"].count("{}") == 1]
        
        if len(requests) == 0:
            return self.base_lm
        
        for _r_idx, _r in enumerate(requests):
            if not _r["prompt"].count("{}") == 1 and _r["prompt"] == "The manager or director position is held by whom?" and _r["subject"]=="Essanay Studios":
                requests[_r_idx]["prompt"] = "The manager or director position in {} is held by whom?"
            if not _r["prompt"].count("{}") == 1 and _r["prompt"] == "What broadcast over radio, TV or the Internet?":
                requests[_r_idx]["prompt"] = "What broadcast {} over radio, TV or the Internet?"

            assert _r["prompt"].count("{}") == 1

        new_lm, _ = apply_memit_to_model(
                    self.base_lm,
                    self.tokenizer,
                    requests,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    #**args_conserve_memory,
                    #**etc_args,
                )

        return new_lm
    
    def reload_lm(self, base_lm):
        self.base_lm = base_lm

# write a new wrapper for your editing methods
'''
class NovelEditor:
    def __init__(self, config, base_lm, tokenizer):

    def edit_lm(self, fact_data, rule_data=None, **kwargs):
        return new_lm

    def reload_lm(self, base_lm):
'''

def prepare_edit_method(config, base_lm, tokenizer):
    def _parse_edit_methods(edit_method_str):
        if edit_method_str == 'mend':
            return MendEditor(config, base_lm, tokenizer)
        elif edit_method_str == 'naive':
            return NaiveEditor(base_lm)
        elif edit_method_str == 'ft':
            if config.eval.lm_type == 'gpt':
                return GPTFTEditor(config, base_lm, tokenizer)
            elif config.eval.lm_type == 'bart':
                return BartFTEditor(config, base_lm, tokenizer)
            else:
                raise ValueError
        elif edit_method_str == 'memit':
            return MemitEditor(config, base_lm, tokenizer)
        else:
            raise ValueError

    editor_estab = _parse_edit_methods(config.eval.estab_method)
    if config.eval.estab_method == config.eval.update_method:
        return {
                "estab": editor_estab,
                "update": editor_estab
            }
    else:
        editor_update = _parse_edit_methods(config.eval.update_method)
        return {
            "estab": editor_estab,
            'update': editor_update
        }
