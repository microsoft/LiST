# LiST ()


This is the implementation of the paper LiST: Lite Self-training Makes Efficient Few-shot Learners. LiST is short for Lite Self-Training.
## Requirements

To run our code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```

**NOTE**: Different versions of packages (like `pytorch`, `transformers`, etc.) may lead to different results from the paper. However, the trend should still hold no matter what versions of packages you use.

## Prepare the data

We download the original datasets (SST-2, MPQA, Subj, MNLI, RTE, QQP) [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). Please run the following commands to prepare data for experiments:

```bash
cd data
bash prepare_dataset.sh
cd ..
```


## Run the model

We prepare scripts to run tasks. Please use bash script under LiST directory. 


Run LiST  as:

```bash
bash run.sh
```

### Notes and Acknowledgments
The implementation is based on https://github.com/huggingface/transformers
We also used some code from: https://github.com/princeton-nlp/LM-BFF
