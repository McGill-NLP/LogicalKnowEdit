### Download

#### DepEdit dataset
The preprocessed data can be downloaded from [Google Drive link](https://drive.google.com/file/d/1sGasr5ffklLJmyN_Yw3pmt15BqD9MF5a/view?usp=sharing).

### Format

The data is in json format and it has already prepared all the QA pairs in the both phases for specific facts and implications. 

* `init`: the knowledge set for establish phase, including `facts` and `rule`. The derived implications are under `queries`.
* `0`, `1`, `2`: each for a copy of establihsed model during update phase
* `score`: the scores given from the annotator.
