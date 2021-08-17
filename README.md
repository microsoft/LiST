# LM-BFF (**B**etter **F**ew-shot **F**ine-tuning of **L**anguage **M**odels)

This is the implementation of the paper [Making Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/pdf/2012.15723.pdf). LM-BFF is short for **b**etter **f**ew-shot **f**ine-tuning of **l**anguage **m**odels.

## Quick links

* [Overview](#overview)
* [Requirements](#requirements)
* [Prepare the data](#prepare-the-data)
* [Run the model](#run-lm-bff)
  * [Quick start](#quick-start)
* [Citation](#citation)


## Overview

![](./figs/lmbff.png)

In this work we present LM-BFF, a suite of simple and complementary techniques for fine-tuning pre-trained language models on a small number of training examples. Our approach includes:

1. Prompt-based fine-tuning together with a novel pipeline for automating prompt generation.
2. A refined strategy for incorporating demonstrations into context.

You can find more details of this work in our [paper](https://arxiv.org/pdf/2012.15723.pdf).

## Requirements

To run our code, please install all the dependency packages by using the following command and update the transformers package to include Deberta model:

```
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers
```

**NOTE**: Different versions of packages (like `pytorch`, `transformers`, etc.) may lead to different results from the paper. However, the trend should still hold no matter what versions of packages you use.

## Prepare the data

We pack the original datasets (SST-2, SST-5, MR, CR, MPQA, Subj, TREC, CoLA, MNLI, SNLI, QNLI, RTE, MRPC, QQP, STS-B) [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). Please download it and extract the files to `./data/original`, or run the following commands:

```bash
cd data
bash download_dataset.sh
```

Then use the following command (in the root directory) to generate the few-shot data we need:

```bash
python tools/generate_k_shot_data.py 
```

See `tools/generate_k_shot_data.py` for more options. For results in the paper, we use the default options: we take `K=16` and take 5 different seeds of 13, 21, 42, 87, 100. The few-shot data will be generated to `data/k-shot`. In the directory of each dataset, there will be folders named as `$K-$SEED` indicating different dataset samples. You can use the following command to check whether the generated data are exactly the same as ours:

```bash
cd data/k-shot
md5sum -c checksum
```

**NOTE**: During training, the model will generate/load cache files in the data folder. If your data have changed, make sure to clean all the cache files (starting with "cache").

## Run LM-BFF

### Quick start
Our code is built on [transformers](https://github.com/huggingface/transformers). 

Most arguments are inherited from `transformers` and are easy to understand. We further explain some of the LM-BFF's arguments:

* `few_shot_type`: There are three modes
  * `finetune`: Standard fine-tuning
  * `prompt`: Prompt-based fine-tuning.
  * `prompt-demo`: Prompt-based fine-tuning with demonstrations.
* `num_k`: Number of training instances for each class. We take `num_k`=16 in our paper. This argument is mainly used for indexing logs afterwards (because the training example numbers are actually decided by the data split you use).
* `template`: Template for prompt-based fine-tuning. We will introduce the template format later.
* `mapping`: Label word mapping for prompt-based fine-tuning. It is a string of dictionary indicating the mapping from label names to label words. **NOTE**: For RoBERTa, the model will automatically add space before the word. See the paper appendix for details.
* `num_sample`: When using demonstrations during inference, the number of samples for each input query. Say `num_sample`=16, then we sample 16 different sets of demonstrations for one input, do the forward seperately, and average the logits for all 16 samples as the final prediction.
* `CT`: Contrasting learning. CT=0 indicates no contrasting learning and 1 indicates contrasting learning for labeled data, 2 indicates contrasting learning for pseudo-labeled data.
* `OPT`: Unlabled data selection option. confidence means filter unlabeled data based on confidence values and meta_st is based meat-learning.
* `SOFT_LABEL`: Use soft label or not. 1 indicates using soft labeles
* `SEMI`:Use semi-supervised learning or not.
* `ADV`: Option for adversarial training. 0 indicates no adv training, 1 is using adv training for labeled data and 2 is using adv training for unlabeled data with data selection, 3 is using adv training for both of labeled data and unlabled data.
* `hybrid`: Option for using CLS and Prompt-based together. 0 is to turn off hybrid training. 

To easily run our experiments, you can also use `run_MNLI.sh` (this command runs prompt-based fine-tuning with demonstrations, no filtering, manual prompt):

```bash
TAG=exp TYPE=prompt-demo TASK=MNLI BS=4 LR=1e-5  MODEL=roberta-base CT=0 hybrid=0 OPT=confidence SOFT_LABEL=0 SEMI=0 ADV=1 GPU=0 bash run_MNLI.sh
```


## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Tianyu (`tianyug@cs.princeton.edu`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use LM-BFF in your work:

```
@article{gao2020making,
  title={Making Pre-trained Language Models Better Few-shot Learners},
  author={Gao, Tianyu and Fisch, Adam and Chen, Danqi},
  journal={arXiv preprint arXiv:2012.15723},
  year={2020}
}
```
