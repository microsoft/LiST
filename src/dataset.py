"""Dataset utils for different data settings for GLUE."""

import os
import copy
import logging
import torch
import numpy as np
import time
from filelock import FileLock
import json
import itertools
import random
import transformers
from processors import processors_mapping, CLUE_processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, median_mapping
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import pandas as pd
from torch.utils import data
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class OurInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None # Position of the mask token
    label_word_list: Optional[List[int]] = None # Label word mapping (dynamic)
    block_flag: Optional[List[int]] = None
    example_id: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

def input_example_to_string(example, sep_token):
    if example.text_b is None:
        return example.text_a
    else:
        # Warning: very simple hack here
        return example.text_a + ' ' + sep_token + ' ' + example.text_b

def input_example_to_tuple(example):
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]

def tokenize_multipart_input(
    input_text_list,
    max_length,
    tokenizer,
    task_name=None,
    prompt=False,
    template=None,
    label_word_list=None,
    first_sent_limit=None,
    other_sent_limit=None,
    gpt3=False,
    truncate_head=False,
    support_labels=None,
    continuous_prompt=0,
    continuous_prompt_length=1,
    example_id=0
):
    
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids = []
    attention_mask = []
    token_type_ids = [] # Only for BERT
    mask_pos = None # Position of the mask token

    if 't5' in type(tokenizer).__name__.lower():
        special_token_mapping = {
            'cls': 3, 'mask': 32099, 'sep': tokenizer.eos_token_id,
            'sep+': tokenizer.eos_token_id,
            'pseudo_token': tokenizer.unk_token_id
        }

    else:
        special_token_mapping = {
            'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id,
            'sep+': tokenizer.sep_token_id,
            'pseudo_token': tokenizer.unk_token_id
        }

    if prompt:
        """
        Concatenate all sentences and prompts based on the provided template.
        Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
        *xx* represent variables:
            *cls*: cls_token
            *mask*: mask_token
            *sep*: sep_token
            *sep+*: sep_token, also means +1 for segment id
            *sent_i*: sentence i (input_text_list[i])
            *sent-_i*: same as above, but delete the last token
            *sentl_i*: same as above, but use lower case for the first word
            *sentl-_i*: same as above, but use lower case for the first word and delete the last token
            *+sent_i*: same as above, but add a space before the sentence
            *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
            *label_i*: label_word_list[i]
            *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning

        Use "_" to replace space.
        PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
        """
        assert template is not None

        if  't5' in type(tokenizer).__name__.lower():
            special_token_mapping = {
                'cls': 3, 'mask': 32099, 'sep': tokenizer.eos_token_id,
                'sep+': tokenizer.eos_token_id,
                'pseudo_token': tokenizer.unk_token_id
            }

        else:
            special_token_mapping = {
                'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id,
                'pseudo_token': tokenizer.unk_token_id
            }

        template_list = template.split('*') # Get variable list in the template
        segment_id = 0 # Current segment id. Segment id +1 if encountering sep+.
        block_flag = []

        first_sep = True
        for part_id, part in enumerate(template_list):
            new_tokens = []

            segment_plus_1_flag = False
            if part in special_token_mapping:
                # if part == 'cls' and 'T5' in type(tokenizer).__name__:
                #     # T5 does not have cls token
                #     continue
                new_tokens.append(special_token_mapping[part])
                if part == 'sep+':
                    segment_plus_1_flag = True
            elif part[:6] == 'label_':
                # Note that label_word_list already has extra space, so do not add more space ahead of it.
                label_id = int(part.split('_')[1])
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:7] == 'labelx_':
                instance_id = int(part.split('_')[1])
                label_id = support_labels[instance_id]
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:5] == 'sent_':
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id])
            elif part[:6] == '+sent_':
                # Add space
                sent_id = int(part.split('_')[1])
                new_tokens += enc(' ' + input_text_list[sent_id])
            elif part[:6] == 'sent-_':
                # Delete the last token
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id][:-1])
            elif part[:6] == 'sentl_':
                # Lower case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentl_':
                # Lower case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(' ' + text)
            elif part[:7] == 'sentl-_':
                # Lower case the first token and discard the last token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text[:-1])
            elif part[:6] == 'sentu_':
                # Upper case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentu_':
                # Upper case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(' ' + text)
            else:
                # Just natural language prompt
                part = part.replace('_', ' ')
                # handle special case when T5 tokenizer might add an extra space
                if len(part) == 1:
                    new_tokens.append(tokenizer.convert_tokens_to_ids(part))
                else:
                    new_tokens += enc(part)

            if part[:4] == 'sent' or part[1:5] == 'sent':
                # If this part is the sentence, limit the sentence length
                sent_id = int(part.split('_')[1])
                if sent_id == 0:
                    if first_sent_limit is not None:
                        new_tokens = new_tokens[:first_sent_limit]
                else:
                    if other_sent_limit is not None:
                        new_tokens = new_tokens[:other_sent_limit]

            block_flag += [0 for i in range(len(new_tokens))]

            input_ids += new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
            token_type_ids += [segment_id for i in range(len(new_tokens))]


            if segment_plus_1_flag:
                segment_id += 1
    else:
        input_ids = [tokenizer.cls_token_id]
        attention_mask = [1]
        token_type_ids = [0]
        block_flag = [0]

        for sent_id, input_text in enumerate(input_text_list):
            if input_text is None:
                # Do not have text_b
                continue
            if pd.isna(input_text) or input_text is None:
                # Empty input
                input_text = ''
            input_tokens = enc(input_text) + [tokenizer.sep_token_id]
            input_ids += input_tokens
            attention_mask += [1 for i in range(len(input_tokens))]
            block_flag += [0 for i in range(len(input_tokens))]
            token_type_ids += [sent_id for i in range(len(input_tokens))]

        if 'T5' in type(tokenizer).__name__: # T5 does not have CLS token
            input_ids = input_ids[1:]
            attention_mask = attention_mask[1:]
            token_type_ids = token_type_ids[1:]
            block_flag = block_flag[1:]



    # Padding
    if first_sent_limit is not None and len(input_ids) > max_length:
        # If using sentence limit, the total length still exceeds the maximum limit, report a warning
        logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))

    if continuous_prompt == 1:
        block_max_length = max_length - continuous_prompt_length - 2

    else:
        block_max_length = max_length - 2


    original_input_ids = input_ids

    if len(input_ids) > block_max_length:

        if prompt:

            mask_pos = input_ids.index(special_token_mapping['mask'])

        else:
            mask_pos = 0

        if mask_pos < block_max_length:
            input_ids = input_ids[:block_max_length]
            attention_mask = attention_mask[:block_max_length]
            token_type_ids = token_type_ids[:block_max_length]
            block_flag = block_flag[:block_max_length]

            input_ids.append(special_token_mapping['sep'])
            attention_mask.append(1)
            token_type_ids.append(token_type_ids[-1])
            block_flag.append(0)
        else:
            remain_len = len(input_ids) - mask_pos
            if remain_len > int(block_max_length / 2):
                start_token_pos = mask_pos - int(block_max_length / 2)
                end_token_pos = mask_pos + int(block_max_length / 2)

                input_ids[start_token_pos] = input_ids[0]
                input_ids[end_token_pos - 1] = input_ids[-1]

                input_ids = input_ids[start_token_pos:end_token_pos]
                attention_mask = attention_mask[start_token_pos:end_token_pos]
                token_type_ids = token_type_ids[start_token_pos:end_token_pos]
                block_flag = block_flag[start_token_pos:end_token_pos]
                input_ids.append(special_token_mapping['sep'])
                attention_mask.append(1)
                token_type_ids.append(token_type_ids[-1])
                block_flag.append(0)

            else:
                input_ids = input_ids[-block_max_length:]
                attention_mask = attention_mask[-block_max_length:]
                token_type_ids = token_type_ids[-block_max_length:]
                block_flag = block_flag[-block_max_length:]




    token_type_id = token_type_ids[-1] + 1

    if continuous_prompt == 1:

        for _ in range(continuous_prompt_length):
            input_ids.append(special_token_mapping['pseudo_token'])
            attention_mask.append(1)
            token_type_ids.append(token_type_id)
            block_flag.append(1)


        input_ids.append(special_token_mapping['sep'])
        attention_mask.append(1)
        token_type_ids.append(token_type_id)
        block_flag.append(0)


    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)
        block_flag.append(0)




    assert  len(input_ids) == max_length, "sequence length is incorrect"
    assert  len(attention_mask) == max_length, "sequence length is incorrect"
    assert  len(token_type_ids) == max_length, "sequence length is incorrect"
    assert  len(block_flag) == max_length, "sequence length is incorrect"


    #
    #
    # if len(input_ids) > max_length:
    #
    #     mask_pos = input_ids.index(tokenizer.mask_token_id)
    #
    #     if mask_pos < max_length:
    #         input_ids = input_ids[:max_length]
    #         attention_mask = attention_mask[:max_length]
    #         token_type_ids = token_type_ids[:max_length]
    #         block_flag = block_flag[:max_length]
    #     else:
    #         remain_len = len(input_ids) - mask_pos
    #         if remain_len > int(max_length/2):
    #             start_token_pos = mask_pos - int(max_length / 2)
    #             end_token_pos = mask_pos + int(max_length / 2)
    #
    #             input_ids[start_token_pos] = input_ids[0]
    #             input_ids[end_token_pos-1] = input_ids[-1]
    #
    #             input_ids = input_ids[start_token_pos:end_token_pos]
    #             attention_mask = attention_mask[start_token_pos:end_token_pos]
    #             token_type_ids = token_type_ids[start_token_pos:end_token_pos]
    #             block_flag = block_flag[start_token_pos:end_token_pos]
    #
    #         else:
    #             input_ids = input_ids[-max_length:]
    #             attention_mask = attention_mask[-max_length:]
    #             token_type_ids = token_type_ids[-max_length:]
    #             block_flag = block_flag[-max_length:]



            #
            #
            # if truncate_head:
            #     input_ids = input_ids[-first_reduce_token_length:]
            #     attention_mask = attention_mask[-first_reduce_token_length:]
            #     token_type_ids = token_type_ids[-max_lengtfirst_reduce_token_length:]
            #     block_flag = block_flag[-first_reduce_token_length:]



            # truncate_head = True


        # if truncate_head:
        #     input_ids = input_ids[-max_length:]
        #     attention_mask = attention_mask[-max_length:]
        #     token_type_ids = token_type_ids[-max_length:]
        #     block_flag = block_flag[-max_length:]
        #
        #
        # else:
        # Default is to truncate the tail




    # input_ids.append(tokenizer.pad_token_id)
    # attention_mask.append(0)
    # token_type_ids.append(0)

    # Find mask token
    if prompt:
        try:
            mask_pos = [input_ids.index(special_token_mapping['mask'])]
        except:
            import pdb
            pdb.set_trace()
        # Make sure that the masked position is inside the max_length

            assert mask_pos[0] < max_length

    result = {'input_ids': input_ids, 'attention_mask': attention_mask}

    if 'Bert' in type(tokenizer).__name__ or ('deberta' in type(tokenizer).__name__.lower()):
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids



    if continuous_prompt == 1:
        result['block_flag'] = block_flag




    if prompt:
        result['mask_pos'] = mask_pos
        result['example_id'] = [example_id]
    # if continuous_prompt:
    #     result['continuous_prompt_mask'] =  \


    for l in result:
        if result[l] is None:
            import pdb
            pdb.set_trace()


    return result



class FewShotDataset(torch.utils.data.Dataset):
    """Few-shot dataset."""

    def __init__(self, args, tokenizer, cache_dir=None, mode="train", use_demo=False, continuous_prompt=None):
        self.args = args
        self.task_name = args.task_name

        if self.args.use_clue:
            self.processor = CLUE_processors_mapping[args.task_name]
        else:
            self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer
        self.mode = mode
        if continuous_prompt is not None:
            self.continuous_prompt = continuous_prompt
        else:
            self.continuous_prompt = args.continuous_prompt

        # If not using demonstrations, use use_demo=True
        self.use_demo = use_demo
        if self.use_demo:
            logger.info("Use demonstrations")
        assert mode in ["train", "un_train", "dev", "test", 'train_large']

        # Get label list and (for prompt) label word list
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        if args.prompt:
            assert args.mapping is not None
            self.label_to_word = eval(args.mapping)

            for key in self.label_to_word:
                # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
                if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                    # Make sure space+word is in the vocabulary
                    assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
                    self.label_to_word[key] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + self.label_to_word[key])[0])
                else:
                    self.label_to_word[key] = tokenizer.convert_tokens_to_ids(self.label_to_word[key])
                logger.info("Label {} to word {} ({})".format(key, tokenizer._convert_id_to_token(self.label_to_word[key]), self.label_to_word[key]))

            if len(self.label_list) > 1:
                self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                # Regression task
                # '0' represents low polarity and '1' represents high polarity.
                self.label_word_list = [self.label_to_word[label] for label in ['0', '1']]
        else:
            self.label_to_word = None
            self.label_word_list = None

        # Multiple sampling: when using demonstrations, we sample different combinations of demonstrations during
        # inference and aggregate the results by averaging the logits. The number of different samples is num_sample.
        if (mode == "train") or not self.use_demo:
            # We do not do multiple sampling when not using demonstrations or when it's the training mode
            self.num_sample = 1
        else:
            self.num_sample = args.num_sample

        # If we use multiple templates, we also need to do multiple sampling during inference.
        if args.prompt and args.template_list is not None:
            logger.info("There are %d templates. Multiply num_sample by %d" % (len(args.template_list), len(args.template_list)))
            self.num_sample *= len(args.template_list)

        logger.info("Total num_sample for mode %s: %d" % (mode, self.num_sample))

        # Load cache
        # Cache name distinguishes mode, task name, tokenizer, and length. So if you change anything beyond these elements, make sure to clear your cache.
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )

        logger.info(f"Creating/loading examples from dataset file at {args.data_dir}")

        lock_path = cached_features_file + ".lock"



        if args.write_cache:

             with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not args.overwrite_cache:
                    start = time.time()
                    self.support_examples, self.query_examples = torch.load(cached_features_file)
                    logger.info(
                        f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                    )
                else:
                    logger.info(f"Creating features from dataset file at {args.data_dir}")

                    # The support examples are sourced from the training set.
                    try:
                        self.support_examples = self.processor.get_train_examples(args.data_dir)
                    except:
                        logger.info("Change to a different processer")
                        self.processor = processors_mapping[args.task_name]
                        self.support_examples = self.processor.get_train_examples(args.data_dir)

                    self.processor.support_examples = self.support_examples

                    if mode == "un_train":
                        self.query_examples = self.processor.get_un_train_examples(args.data_dir)
                    elif mode == "dev":
                        self.query_examples = self.processor.get_dev_examples(args.data_dir)
                    elif mode == "test":
                        self.query_examples = self.processor.get_test_examples(args.data_dir)
                    else:
                        self.query_examples = self.support_examples

                    start = time.time()
                    torch.save([self.support_examples, self.query_examples], cached_features_file)
                    # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                    logger.info(
                        "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )
        else:
            self.support_examples = self.processor.get_train_examples(args.data_dir)
            self.processor.support_examples = self.support_examples

            if mode == "un_train":
                self.query_examples = self.processor.get_un_train_examples(args.data_dir)
            elif mode == "train_large":
                self.query_examples = self.processor.get_large_train_examples(args.data_dir)
            elif mode == "dev":
                self.query_examples = self.processor.get_dev_examples(args.data_dir)
            elif mode == "test":
                self.query_examples = self.processor.get_test_examples(args.data_dir)
            else:
                self.query_examples = self.support_examples


        # For filtering in using demonstrations, load pre-calculated embeddings
        if self.use_demo and args.demo_filter:
            split_name = ''
            if mode == 'train':
                split_name = 'train'
            elif mode == 'un_train':
                split_name = 'un_train'
            elif mode == 'dev':
                if args.task_name == 'mnli':
                    split_name = 'dev_matched'
                elif args.task_name == 'mnli-mm':
                    split_name = 'dev_mismatched'
                else:
                    split_name = 'dev'
            elif mode == 'test':
                if args.task_name == 'mnli':
                    split_name = 'test_matched'
                elif args.task_name == 'mnli-mm':
                    split_name = 'test_mismatched'
                else:
                    split_name = 'test'
            else:
                raise NotImplementedError

            self.support_emb = np.load(os.path.join(args.data_dir, "train_{}.npy".format(args.demo_filter_model)))
            self.query_emb = np.load(os.path.join(args.data_dir, "{}_{}.npy".format(split_name, args.demo_filter_model)))
            logger.info("Load embeddings (for demonstration filtering) from {}".format(os.path.join(args.data_dir, "{}_{}.npy".format(split_name, args.demo_filter_model))))

            assert len(self.support_emb) == len(self.support_examples)
            assert len(self.query_emb) == len(self.query_examples)

        # Size is expanded by num_sample
        self.size = len(self.query_examples) * self.num_sample

        # Prepare examples (especially for using demonstrations)
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for sample_idx in range(self.num_sample):
            for query_idx in range(len(self.query_examples)):
                # If training, exclude the current example. Else keep all.
                if self.use_demo and args.demo_filter:
                    # Demonstration filtering
                    candidate = [support_idx for support_idx in support_indices
                                   if support_idx != query_idx or mode != "train"]
                    sim_score = []
                    for support_idx in candidate:
                        sim_score.append((support_idx, util.pytorch_cos_sim(self.support_emb[support_idx], self.query_emb[query_idx])))
                    sim_score.sort(key=lambda x: x[1], reverse=True)
                    if self.num_labels == 1:
                        # Regression task
                        limit_each_label = int(len(sim_score) // 2 * args.demo_filter_rate)
                        count_each_label = {'0': 0, '1': 0}
                        context_indices = []

                        if args.debug_mode:
                            print("Query %s: %s" % (self.query_examples[query_idx].label, self.query_examples[query_idx].text_a)) # debug
                        for support_idx, score in sim_score:
                            if count_each_label['0' if float(self.support_examples[support_idx].label) <= median_mapping[args.task_name] else '1'] < limit_each_label:
                                count_each_label['0' if float(self.support_examples[support_idx].label) <= median_mapping[args.task_name] else '1'] += 1
                                context_indices.append(support_idx)
                                if args.debug_mode:
                                    print("    %.4f %s | %s" % (score, self.support_examples[support_idx].label, self.support_examples[support_idx].text_a)) # debug
                    else:
                        limit_each_label = int(len(sim_score) // self.num_labels * args.demo_filter_rate)
                        count_each_label = {label: 0 for label in self.label_list}
                        context_indices = []

                        if args.debug_mode:
                            print("Query %s: %s" % (self.query_examples[query_idx].label, self.query_examples[query_idx].text_a)) # debug
                        for support_idx, score in sim_score:
                            if count_each_label[self.support_examples[support_idx].label] < limit_each_label:
                                count_each_label[self.support_examples[support_idx].label] += 1
                                context_indices.append(support_idx)
                                if args.debug_mode:
                                    print("    %.4f %s | %s" % (score, self.support_examples[support_idx].label, self.support_examples[support_idx].text_a)) # debug
                else:

                    # Using demonstrations without filtering
                    if self.use_demo:
                        if mode == 'train':
                            context_indices = list(set(support_indices) - set([query_idx]))
                        else:
                            context_indices = support_indices
                    else:
                        context_indices=[]

                # We'll subsample context_indices further later.
                self.example_idx.append((query_idx, context_indices, sample_idx))

        # If it is not training, we pre-process the data; otherwise, we process the data online.

        if mode not in ["train", "un_train"]:
            self.features = []
            _ = 0
            for query_idx, context_indices, bootstrap_idx in self.example_idx:
                # The input (query) example
                example = self.query_examples[query_idx]
                # The demonstrations
                supports = self.select_context([self.support_examples[i] for i in context_indices])

                if args.template_list is not None:
                    template = args.template_list[sample_idx % len(args.template_list)] # Use template in order
                else:
                    template = args.template

                self.features.append(self.convert_fn(
                    example=example,
                    supports=supports,
                    use_demo=self.use_demo,
                    label_list=self.label_list,
                    prompt=args.prompt,
                    template=template,
                    label_word_list=self.label_word_list,
                    verbose=True if _ == 0 else False,
                    example_id=query_idx

                ))

                _ += 1
        else:
            self.features = None

    def select_context(self, context_examples):
        """
        Select demonstrations from provided examples.
        """
        max_demo_per_label = 1
        counts = {k: 0 for k in self.label_list}
        if len(self.label_list) == 1:
            # Regression
            counts = {'0': 0, '1': 0}
        selection = []

        if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
            # For GPT-3's in-context learning, we sample gpt3_in_context_num demonstrations randomly.
            order = np.random.permutation(len(context_examples))
            for i in range(min(self.args.gpt3_in_context_num, len(order))):
                selection.append(context_examples[order[i]])
        else:
            # Our sampling strategy
            order = np.random.permutation(len(context_examples))

            for i in order:
                label = context_examples[i].label
                if len(self.label_list) == 1:
                    # Regression
                    label = '0' if float(label) <= median_mapping[self.args.task_name] else '1'
                if counts[label] < max_demo_per_label:
                    selection.append(context_examples[i])
                    counts[label] += 1
                if sum(counts.values()) == len(counts) * max_demo_per_label:
                    break


        return selection

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.features is None:
            query_idx, context_indices, bootstrap_idx = self.example_idx[i]
            # The input (query) example
            example = self.query_examples[query_idx]
            # The demonstrations
            supports = self.select_context([self.support_examples[j] for j in context_indices])

            if self.args.template_list is not None:
                template = self.args.template_list[sample_idx % len(self.args.template_list)]
            else:
                template = self.args.template


            features = self.convert_fn(
                example=example,
                supports=supports,
                use_demo=self.use_demo,
                label_list=self.label_list,
                prompt=self.args.prompt,
                template=template,
                label_word_list=self.label_word_list,
                verbose=False,
                example_id=query_idx,
            )

        else:
            features = self.features[i]

        return features

    def get_labels(self):
        return self.label_list


    def convert_fn(
        self,
        example,
        supports,
        use_demo=False,
        label_list=None,
        prompt=False,
        template=None,
        label_word_list=None,
        verbose=False,
        example_id=None
    ):
        """
        Returns a list of processed "InputFeatures".
        """
        max_length = self.args.max_seq_length

        # Prepare labels
        label_map = {label: i for i, label in enumerate(label_list)} # Mapping the label names to label ids
        if len(label_list) == 1:
            # Regression
            label_map = {'0': 0, '1': 1}

        # Get example's label id (for training/inference)
        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            # Regerssion
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]

        # Prepare other features
        if not use_demo:
            # No using demonstrations
            inputs = tokenize_multipart_input(
                input_text_list=input_example_to_tuple(example),
                max_length=max_length,
                tokenizer=self.tokenizer,
                task_name=self.args.task_name,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
                continuous_prompt=self.continuous_prompt,
                continuous_prompt_length=self.args.prompt_length,
                example_id=example_id
            )
            features = OurInputFeatures(**inputs, label=example_label)

        else:
            # Using demonstrations

            # Max length
            if self.args.double_demo:
                # When using demonstrations, double the maximum length
                # Note that in this case, args.max_seq_length is the maximum length for a single sentence
                max_length = max_length * 2
            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                # When using GPT-3's in-context learning, take the maximum tokenization length of the model (512)
                max_length = 512

            # All input sentences, including the query and the demonstrations, are put into augmented_examples,
            # and are numbered based on the order (starting from 0). For single sentence tasks, the input (query)
            # is the sentence 0; for sentence-pair tasks, the input (query) is the sentence 0 and 1. Note that for GPT-3's
            # in-context learning, the input (query) might be at the end instead of the beginning (gpt3_in_context_head)
            augmented_example = []
            query_text = input_example_to_tuple(example) # Input sentence list for query
            support_by_label = [[] for i in range(len(label_map))]

            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                support_labels = []
                augmented_example = query_text
                for support_example in supports:
                    augmented_example += input_example_to_tuple(support_example)
                    current_label = support_example.label
                    if len(label_list) == 1:
                        current_label = '0' if float(current_label) <= median_mapping[self.args.task_name] else '1' # Regression
                    support_labels.append(label_map[current_label])
            else:
                # Group support examples by label
                for label_name, label_id in label_map.items():
                    if len(label_list) == 1:
                        # Regression
                        for support_example in filter(lambda s: ('0' if float(s.label) <= median_mapping[self.args.task_name] else '1') == label_name, supports):
                            support_by_label[label_id] += input_example_to_tuple(support_example)
                    else:
                        for support_example in filter(lambda s: s.label == label_name, supports):
                            support_by_label[label_id] += input_example_to_tuple(support_example)

                augmented_example = query_text
                for label_id in range(len(label_map)):
                    augmented_example += support_by_label[label_id]

            # Tokenization (based on the template)
            inputs = tokenize_multipart_input(
                input_text_list=augmented_example,
                max_length=max_length,
                tokenizer=self.tokenizer,
                task_name=self.args.task_name,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
                truncate_head=self.args.truncate_head,
                gpt3=self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail,
                support_labels=None if not (self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail) else support_labels,
                continuous_prompt=self.continuous_prompt,
                continuous_prompt_length=self.args.prompt_length,
                example_id= example_id
            )
            features = OurInputFeatures(**inputs, label=example_label)

        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features)
            logger.info("text: %s" % self.tokenizer.decode(features.input_ids))

        return features

def patch_data(dataset, batch_data):

    indice_list = batch_data['example_id'].cpu().tolist()
    device = batch_data['example_id'].device
    new_data_list = [dataset[i[0]] for i in indice_list]
    new_batch_data = {}
    for single_data in new_data_list:
        for key in single_data.__dict__:
            if key not in batch_data:
                continue
            if key not in new_batch_data:
                new_batch_data[key] = []
                new_batch_data[key].append(single_data.__dict__[key])
            else:
                if new_batch_data[key][0] is not None:
                    new_batch_data[key].append(single_data.__dict__[key])
    for key in new_batch_data:
        if new_batch_data[key][0] is not None:

            new_batch_data[key] = torch.Tensor(new_batch_data[key]).type(batch_data[key].type())
            #new_batch_data[key] = new_batch_data[key].to(device)

    return new_batch_data





