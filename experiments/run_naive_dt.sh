python3 $HOME/logicalKnowEdit/evaluation.py ++eval.estab_method="naive" ++eval.update_method="naive"\
        ++eval.metrics=['infer']\
        ++eval.input_path=$HOME/scratch/knowedit/dataset/depedit_dt.jsonl\
        ++eval.lm_type="bart"\
        ++eval.base_lm_path=$HOME/scratch/knowedit/models/bart-base/\
        ++eval.bart_qa_path=$HOME/scratch/knowedit/models/QA_model.ckpt 
