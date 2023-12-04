python3 $HOME/standup_rf/evaluation.py ++eval.estab_method="memit" ++eval.update_method="memit"\
        ++eval.input_path=$HOME/scratch/knowedit/standup_data/depedit_dt.jsonl\
        ++eval.lm_type="gpt" ++eval.max_ft_steps=10\
        ++eval.base_lm_path=$HOME/scratch/knowedit/models_gpt/gpt2xl_ft/best_tfmr/\
        ++eval.memit_params_path=$HOME/standup_rf/memit/hparams/MEMIT/gpt2-xl.json\
        ++eval.memit_cache_template_dir=$HOME/scratch/knowedit/gpt-xl/