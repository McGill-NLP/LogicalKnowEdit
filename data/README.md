
####  DepEdit dataset

The preprocessed data can be downloaded from [this link](https://drive.google.com/file/d/1qVBgekrHZnPNyWuMkphtIaamESunrd1B/view?usp=sharing).

In the original experiments, the unrelated facts are sampled on the fly. You can also use [this version](https://drive.google.com/file/d/1H11T5n-ZXiwpR9lcYiuIkd6uAXatQsSt/view?usp=sharing), where the unrelated facts are sampled beforehand. If so, setting `do_sample_irre=False` in `config_eval.yaml`.

####  Format

The data is in json format and it has already preprocessed all the QA pairs of specific facts and implications for both phases. The schema is explained as below. `init` is for the establish phase, while `0`, `1`, `2` are for the update phase.

```javascript
{
    "init": {
        ...
    },
    "0": {
        "facts": [
            {
                "q": query,
                "a": answer,
                "trips": (subject, relation, object),
                "is_update": the fact is updated
            }
        ],
        "rule": {
            "pre1": premise-1 template,
            "pre2": premise-2 template,
            "imp": implication template
        },
        "queries": {
            "original": {
                "facts": [
                    {
                    "q" the same query,
                    "a": answer
                    },
                    ...
                ],
                "inference": [
                    {
                    "q" query of implications,
                    "a": answer
                    },
                    ...
                ]
            },
            "semantic-equiv": {
                "facts": [
                    {
                    "q" different query,
                    "a": answer
                    },
                    ...
                ],
                ...
            },
        }
    },
    "1": {
        ...
    },
    "2": {
        ...
    },
    "score": two plausbility scores of the rule
}
```

####  Others
Besides the datasets used for experiments, there are also (1) a larger test set that includes extra rules, rated with scores `[4,5]` (i.e. *must be true* and *likely to be true*) (2) the complete set of rule candidates, along with their corresponding scores provided by the annotators.

If you are interested in accessing either of these resources, please send an email request to zichao.li AT mail.mcgill.ca