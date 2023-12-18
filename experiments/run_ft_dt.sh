python3 $HOME/logicalKnowEdit/evaluation.py ++eval.estab_method="ft" ++eval.update_method="ft"\
        ++eval.metrics=['consistency', 'infer']\
        ++eval.input_path=$HOME/scratch/knowedit/dataset/depedit_dt.jsonl\
        ++eval.lm_type="bart"\
        ++eval.base_lm_path=$HOME/scratch/knowedit/models/bart-base/\
        ++eval.max_ft_step=30\
        ++eval.bart_qa_path=$HOME/scratch/knowedit/models/QA_model.ckpt 
