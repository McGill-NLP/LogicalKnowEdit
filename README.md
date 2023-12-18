### Dependencies

`pip install -r requirements.txt`.

For the experiments in the paper, we evaluate [Mend](https://github.com/eric-mitchell/mend) and [Memit](https://github.com/kmeng01/memit). Therefore, they are imported into `editing.py`.


### Evaluating a new method
To evaluate your custom editing methods:

1. Create a new wrapper in `editing.py` for your methods. Please look at existing wrappers for reference.

2. If you want to edit a new base lm, write a function similar to `prepare_gpt_model` in `eval_utils.py`. Also, update the `./config/config_eval.yaml` file to match your changes.

For practical examples of how to run tests with our dataset, look in the scripts under `/experiments/`.

### Data
Please refer to `/data/` for instructions on how to access and use our dataset.

### Citation

```
@inproceedings{li2023evaluating,
  title={Evaluating Dependencies in Fact Editing for Language Models: Specificity and Implication Awareness},
  author={Li, Zichao and Arous, Ines and Reddy, Siva and Cheung, Jackie Chi Kit},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={7623--7636},
  year={2023}
}
```