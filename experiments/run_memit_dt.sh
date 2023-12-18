python3 $HOME/logicalKnowEdit/evaluation.py ++eval.estab_method="memit" ++eval.update_method="memit"\
        ++eval.metrics=['consistency', 'infer']\
        ++eval.input_path=$HOME/scratch/knowedit/dataset/depedit_dt.jsonl\
        ++eval.lm_type="gpt" ++eval.max_ft_steps=10\
        ++eval.base_lm_path=$HOME/scratch/knowedit/models_gpt/gpt2xl_ft/best_tfmr/\
        ++eval.memit_params_path=$HOME/memit/hparams/MEMIT/gpt2-xl.json\
        ++eval.memit_cache_template_dir=$HOME/scratch/knowedit/gpt-xl/