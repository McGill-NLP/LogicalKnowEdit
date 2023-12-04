python3 $HOME/knowedit_github/evaluation.py ++eval.estab_method="naive" ++eval.update_method="naive"\
        ++eval.input_path=$HOME/scratch/knowedit/standup_data/depedit_dt.jsonl\
        ++eval.lm_type="bart"\
        ++eval.base_lm_path=$HOME/scratch/knowedit/models/bart-base/\
        ++eval.bart_qa_path=$HOME/scratch/knowedit/models/QA_model.ckpt 
