"""Finetuning the library models for sequence classification on GLUE."""

import dataclasses
import logging
import os
import pdb
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import torch

import numpy as np

import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed

from dataset import FewShotDataset
from models import LMForPromptFinetuning, BertForPromptFinetuning, RobertaForPromptFinetuning, DebertaForPromptFinetuning, resize_token_type_embeddings
from trainer import Trainer
from processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping

from filelock import FileLock
from datetime import datetime

from copy import deepcopy
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


MAPPING = {'mnli': "{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}",
           "sst-2": "{'0':'terrible','1':'great'}",
           "qqp": "{'0':'No','1':'Yes'}",
           "rte": "{'not_entailment':'No','entailment':'Yes'}",
           "mpqa": "{0:'terrible',1:'great'}",
           "subj": "{0:'subjective',1:'objective'}",
           "cola": "{'0':'incorrect','1':'correct'}"
           }
TEMPLATE={"mnli":"*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
          'sst-2':"*cls**sent_0*_It_was*mask*.*sep+*",
           "qqp": "*cls**sent_0**mask*,*+sentl_1**sep+*",
           "rte": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
           "mpqa": "*cls**sent_0*_It_was*mask*.*sep+*",
           "subj": "*cls**sent_0*_This_is*mask*.*sep+*",
           "cola":"*cls**sent_0*_This_is*mask*.*sep+*"
          }


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    hybrid: int = field(
        default=0,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )


    adv_training: int = field(
        default=0,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default='prompt-demo',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"}
    )
    random_segment: bool = field(
        default=False,
        metadata={"help": "add random segment"}
    )

    prompt_encoder_type: str = field(
        default='mlp',
        metadata={"help": ""}
    )
    adapter_dim: int = field(
        default= -1,
        metadata={"help": ""}
    )
    adapter_alpha: int = field(
        default= 32,
        metadata={"help": ""}
    )
    adapter_choice: str = field(
        default='none',
        metadata={"help": ""}
    )
    adapter_init_std: float = field(
        default=0.02,
        metadata={"help": ""}
    )




@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=1,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )

    # For prompting
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    template_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )
 
    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"}
    )

    demo_filter_model: str = field(
        default=None,
        metadata={"help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    # GPT-3's in-context learning
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"}
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"}
    )

    gpt3_in_context_num: int = field(
        default=32,
        metadata={"help": "Number of context examples"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: list = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."}
    )
    continuous_prompt: int = field(
        default=1,
        metadata={"help": "Number of context examples"}
    )
    prompt_length: int = field(
        default=20,
        metadata={"help": ""}
    )
    write_cache: bool = field(
        default=False,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )
    use_clue: bool = field(
        default=False,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )





@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )

    update_teacher_steps: int = field(
        default=40,
        metadata={"help": "steps of updating teacher model"}
    )

    finetune_teacher_epoch: int = field(
        default=20,
        metadata={"help": "steps of finetuning teacher model"}
    )

    self_training_start_epoch: int = field(
        default=20,
        metadata={"help": "Self-training beggining step"}
    )
    self_training_start_iter: int = field(
        default=20,
        metadata={"help": "Self-training beggining step"}
    )


    re_init: bool = field(
        default=False,
        metadata={"help": "Re initialize"}
    )

    is_semi: int = field(
        default=1,
        metadata={"help": "Semi-supervised learning or not"}
    )
    use_psuedo_label: int = field(
        default=1,
        metadata={"help": "Semi-supervised learning or not"}
    )

    un_train_batch_size: int = field(
        default=8,
        metadata={"help": "batch_size for unlabled data"}
    )
    meta_train_batch_size: int = field(
        default=4,
        metadata={"help": "batch_size for unlabled data"}
    )
    prompt_learning_rate: float = field(
        default=1e-5,
        metadata={"help": "threshold for data selection"}
    )

    soft_label: int = field(
        default=1,
        metadata={"help": "batch_size for unlabled data"}
    )

    psuedo_selection_opt: str = field(
        default='confidence',
        metadata={"help": "confidence/meta_st"}
    )

    un_gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "batch_size for unlabled data"}
    )


    mean_teacher: bool = field(
        default=False,
        metadata={"help": "update teacher model via mean-teacher"}
    )

    confidence_thresh: float = field(
        default=0.8,
        metadata={"help": "threshold for data selection"}
    )

    temperature: float = field(
        default=1,
        metadata={"help": "temperature value for prediction sharpen"}
    )
    demo_condon: int = field(
        default=0,
        metadata={"help": "temperature value for prediction sharpen"}
    )
    start_from_freeze: bool = field(
        default=True,
        metadata={"help": "temperature value for prediction sharpen"}
    )
    freeze: bool = field(
        default=False,
        metadata={"help": "temperature value for prediction sharpen"}
    )

    sharpen: bool = field(
        default=False,
        metadata={"help": "sharpen the prediction"}
    )

    adv_opt: int = field(
        default=0,
        metadata={"help": "0: no adv, 1: adv for labeled data, 2: adv for unlabeled data, 3: adv for all the data"}
    )
    co_training: int = field(
        default=0,
        metadata={"help": "0: no adv, 1: adv for labeled data, 2: adv for unlabeled data, 3: adv for all the data"}
    )
    contrast_training: int = field(
        default=1,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )
    sampling_steps: int = field(
        default=1,
        metadata={"help": "Number of context examples"}
    )
    demo_condon_steps: int = field(
        default=3000,
        metadata={"help": "Number of context examples"}
    )
    cpu: bool = field(
        default=False,
        metadata={"help": "Number of context examples"}
    )
    use_last_epoch: bool = field(
        default=False,
        metadata={"help": "Number of context examples"}
    )
    freeze_encoder: bool = field(
        default=False,
        metadata={"help": "Number of context examples"}
    )
    only_train_bias: bool = field(
        default=False,
        metadata={"help": "Number of context examples"}
    )
    self_training_session: int = field(
        default=5,
        metadata={"help": "Number of context examples"}
    )
    update_k_layers: int = field(
        default=-1,
        metadata={"help": "Number of context examples"}
    )
    update_component: str = field(
        default='none',
        metadata={"help": "confidence/meta_st"}
    )
    semi_finetune: bool = field(
        default=False,
        metadata={"help": "self-training use finetuning stage"}
    )
    meta_st_warmup: float = field(
        default=0,
        metadata={"help": "warmup_steps"}
    )
    semi_warmup_ratio: float = field(
        default=0,
        metadata={"help": "warmup_steps"}
    )
    semi_learning_rate: float = field(
        default=2e-5,
        metadata={"help":""}
    )





def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))
  
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split('\t')
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id] 
            logger.info("Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping))
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[:data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(data_args.template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))

    if data_args.prompt is not None and data_args.mapping is None:
        data_args.mapping = MAPPING[data_args.task_name]
    if data_args.prompt is not None and data_args.template is None:
        data_args.template = TEMPLATE[data_args.task_name]
    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Automatically generate template for using demonstrations
    if data_args.auto_demo and model_args.few_shot_type == 'prompt-demo':
        # GPT-3's in-context learning
        if data_args.gpt3_in_context_head or data_args.gpt3_in_context_tail: 
            logger.info("Automatically convert the template to GPT-3's in-context learning.")
            assert data_args.template_list is None

            old_template = data_args.template
            new_template = old_template + ''
            old_template = old_template.replace('*cls*', '')
            # Single sentence or sentence pair?
            sent_num = 1
            if "_1" in old_template:
                sent_num = 2
            for instance_id in range(data_args.gpt3_in_context_num):
                sub_template = old_template + ''
                # Replace sent_id
                for sent_id in range(sent_num):
                    sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * instance_id + sent_id))
                # Replace mask
                sub_template = sub_template.replace("*mask*", "*labelx_{}*".format(instance_id))
                if data_args.gpt3_in_context_tail:
                    new_template = new_template + sub_template # Put context at the end
                else:
                    new_template = sub_template + new_template # Put context at the beginning
            logger.info("| {} => {}".format(data_args.template, new_template))
            data_args.template = new_template
        else:
            logger.info("Automatically convert the template to using demonstrations.")
            if data_args.template_list is not None:
                for i in range(len(data_args.template_list)):
                    old_template = data_args.template_list[i]
                    new_template = old_template + ''
                    old_template = old_template.replace('*cls*', '')
                    # Single sentence or sentence pair?
                    sent_num = 1
                    if "_1" in old_template:
                        sent_num = 2
                    for label_id in range(num_labels):
                        sub_template = old_template + ''
                        # Replace sent id
                        for sent_id in range(sent_num):
                            sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * label_id + sent_id))
                        # Replace mask
                        sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                        new_template = new_template + sub_template
                    logger.info("| {} => {}".format(data_args.template_list[i], new_template))
                    data_args.template_list[i] = new_template
            else:
                old_template = data_args.template
                new_template = old_template + ''
                old_template = old_template.replace('*cls*', '')
                # Single sentence or sentence pair?
                sent_num = 1
                if "_1" in old_template:
                    sent_num = 2
                for label_id in range(num_labels):
                    sub_template = old_template + ''
                    # Replace sent id
                    for sent_id in range(sent_num):
                        sub_template = sub_template.replace("_{}".format(sent_id), "_{}".format(sent_num + sent_num * label_id + sent_id))
                    # Replace mask
                    sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                    new_template = new_template + sub_template
                logger.info("| {} => {}".format(data_args.template, new_template))
                data_args.template = new_template

    # Create config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )


    if 'prompt' in model_args.few_shot_type:
        # if config.model_type == 'roberta':
        #     model_fn = RobertaForPromptFinetuning
        # elif config.model_type == 'bert':
        #     model_fn = BertForPromptFinetuning
        # elif config.model_type =='deberta':
        #     model_fn = DebertaForPromptFinetuning
        model_fn = LMForPromptFinetuning
        # else:
        #     raise NotImplementedError
    elif model_args.few_shot_type == 'finetune':
        model_fn = AutoModelForSequenceClassification
    else:
        raise NotImplementedError

    # if training_args.co_training == 1:
    #     if 'prompt'  in model_args.few_shot_type:
    #         co_model_fn = AutoModelForSequenceClassification
    #
    #     elif model_args.few_shot_type == 'finetune':
    #         if config.model_type == 'roberta':
    #             co_model_fn = RobertaForPromptFinetuning
    #         elif config.model_type == 'bert':
    #             co_model_fn = BertForPromptFinetuning
    #         else:
    #             raise NotImplementedError
    #
    #     else:
    #         raise NotImplementedError
    special_tokens = []



    # Create tokenizer
    if 't5' in model_args.model_name_or_path:
        tokenizer = T5Tokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            additional_special_tokens=special_tokens,
            cache_dir=model_args.cache_dir,
        )

    # Get our special datasets.
    train_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type))
    )

    if training_args.demo_condon_steps > 0:

        train_demo_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=True, continuous_prompt=0)
        )
    else:
        train_demo_dataset = train_dataset


    if training_args.is_semi:

        un_train_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="un_train", use_demo=("demo" in model_args.few_shot_type))
        )
    else:
        un_train_dataset = None

    if not data_args.use_clue and training_args.do_eval:
        eval_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="dev", use_demo=("demo" in model_args.few_shot_type))
            if training_args.do_eval
            else None
        )
    else:
        eval_dataset = None
    test_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="test", use_demo=("demo" in model_args.few_shot_type))
        if training_args.do_predict
        else None
    )


    set_seed(training_args.seed)

    if data_args.prompt:
        data_args.label_word_list = train_dataset.label_word_list


    if data_args.prompt:
        model = model_fn(config, model_args, data_args)
    else:
        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )





    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]
            if data_args.use_clue:
                logits = predictions.reshape([test_dataset.num_sample, -1, num_logits])
            else:
                logits = predictions.reshape([eval_dataset.num_sample, -1, num_logits])
            logits = logits.mean(axis=0)
            
            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            if data_args.use_clue:
                label_ids = p.label_ids.reshape([test_dataset.num_sample, -1])
            else:
                label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    if data_args.use_clue:
        clue_data_args = dataclasses.replace(data_args, task_name=data_args.task_name+"-clue")
        eval_dataset = FewShotDataset(clue_data_args, tokenizer=tokenizer, mode="test",
                                      use_demo=('demo' in model_args.few_shot_type))

    if eval_dataset is None:
        eval_dataset = test_dataset


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        train_demo_dataset=train_demo_dataset,
        un_train_dataset=un_train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name)
    )
    trainer.tokenizer = tokenizer

    # if training_args.co_training == 1:
    #     trainer.co_model = co_model

    # Training
    if training_args.do_train:

        trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
        # Use the early stop, so do not save the model in the end (unless specify save_at_last)
        if training_args.save_at_last:
            trainer.save_model(training_args.output_dir)
 
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
            torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))
        
        # Reload the best checkpoint (for eval)

        checkpoint = os.path.join(training_args.output_dir, 'pytorch_model.bin')

        #model = model.from_pretrained(training_args.output_dir)
        if data_args.prompt:
            model.load_model(checkpoint)
        else:
            model = model_fn.from_pretrained(training_args.output_dir)

        model = model.to(training_args.device)
        trainer.model = model
        # if data_args.prompt:
        #     model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
        # if output_modes_mapping[data_args.task_name] == 'regression':
        #     # lower / upper bounds
        #     model.lb, model.ub = bound_mapping[data_args.task_name]
        # model.model_args = model_args
        # model.data_args = data_args
        # model.tokenizer = tokenizer

    # Evaluation
    final_result = {
        'time': str(datetime.today()),
    }

    #model.load_state_dict(torch.load('/home/t-yaqingwang/azure_storage/projects/few_shot/middle_bert_prompt_mnli/pt-results/few_shot-cf6f4f93-middle_bert_prompt_mnli-search_middle_bert_prompt_mnli_Task_MNLI_bert-base-uncased_seed_100_prompt_0_5-44d6f918/pytorch_model.bin'))

    eval_results = {}
    if training_args.do_eval and (not data_args.use_clue):
        logger.info("*** Validate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics 

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[eval_dataset.args.task_name + '_dev_' + key] = value
            eval_results.update(eval_result)

    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]

        if data_args.use_clue:

            clue_data_args = dataclasses.replace(data_args, task_name=data_args.task_name+"-clue")
            test_datasets.append(
                FewShotDataset(clue_data_args, tokenizer=tokenizer, mode="test",
                               use_demo=('demo' in model_args.few_shot_type))
            )

        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                FewShotDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test",
                               use_demo=('demo' in model_args.few_shot_type))
            )


        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[test_dataset.args.task_name + '_test_' + key] = value

                if training_args.save_logit:
                    predictions = output.predictions
                    num_logits = predictions.shape[-1]
                    logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
                    np.save(os.path.join(training_args.save_logit_dir, "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id, training_args.array_id)), logits)

            test_results.update(test_result)

    with FileLock('log.lock'):
        with open(os.path.join(training_args.output_dir, 'log'), 'a') as f:
            final_result.update(vars(model_args))
            final_result.update(vars(training_args))
            final_result.update(vars(data_args))
            if 'evaluation_strategy' in final_result:
                final_result.pop('evaluation_strategy')
            f.write(str(final_result) + '\n')
    
    return eval_results

if __name__ == "__main__":
    main()
