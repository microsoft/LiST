import torch
from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM
from torch.nn import functional as F
import torch
import numpy as np
import random
import os
#import tokenizers
import pdb


def generate_mask_sentence(sentence, mlm_prob):
    labels = torch.ones([1, len(sentence.split())])
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_prob)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = 0  # We only compute loss on masked tokens
    labels = labels.bool().numpy()[0]
    if labels.sum() == 0:
        labels[random.randint(0, len(labels) - 1)] = True
    return labels


label_sen = {'0': 'It is terrible.', '1': 'It is great.'}
special_token_dict = {'roberta-large': '<mask>', 'bert-base-uncased': '[MASK]'}
for model_name in ['roberta-large']:
    if model_name == 'roberta-large':
        model = RobertaForMaskedLM.from_pretrained(model_name)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        model = BertForMaskedLM.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    special_token = special_token_dict[model_name]
    for seed in [100, 13, 21, 42, 87]:
        data_folder = '/home/yaqing/Projects/Few-shot/data/k-shot/SST-2/20-' + str(seed) + '/'

        f = open(os.path.join(data_folder, 'train_few.tsv'))
        f_out = open(os.path.join(data_folder, 'train_' + model_name.split('-')[0] + '_conditional_0.2_1.tsv'), 'w')

        lines = f.readlines()
        sentence_list = []
        for i, l in enumerate(lines):
            if i == 0:
                f_out.write(l)
                continue
            f_out.write(l)
            line = l.replace('\n', '').split('\t')
            sentence_list.append(line)

        new_sentence_list = []
        for i, s in enumerate(sentence_list):
            sentence = s[0]
            #             for _ in range(5):
            #                 mask_bool = generate_mask_sentence(sentence, mlm_prob=0.2)
            #                 new_sentence = [sentence.split()[i] if not mask else special_token for i, mask in enumerate(mask_bool)]
            #                 new_sentence= ' '.join(new_sentence)
            #                 new_sentence = label_sen[s[1]]+' '+new_sentence
            #                 new_sentence_list.append([i, new_sentence])
            # new_sentence_list.append([i, ' '.join(new_sentence)])
            for _ in range(5):
                mask_bool = generate_mask_sentence(sentence, mlm_prob=0.2)
                new_sentence = [sentence.split()[i] if not mask else special_token for i, mask in enumerate(mask_bool)]
                new_sentence = ' '.join(new_sentence)
                new_sentence = label_sen[s[1]] + ' ' + new_sentence
                new_sentence_list.append([i, new_sentence])
                # new_sentence_list.append([i, ' '.join(new_sentence)])
        special_token_id = tokenizer.convert_tokens_to_ids(special_token)

        for sentence_id, input_txt in new_sentence_list:
            # print(sentence_list[sentence_id], input_txt)

            inputs = tokenizer(input_txt, return_tensors='pt')
            pred_mask = inputs['input_ids'] == special_token_id
            pred_mask = pred_mask[0]

            outputs = model(**inputs)
            predictions = outputs[0]
            sorted_preds, sorted_idx = predictions[0].sort(dim=-1, descending=True)
            length = inputs['input_ids'].size(1)
            for k in range(1):
                predicted_index = [sorted_idx[i, k].item() for i in range(0, length)]
                original_index = inputs['input_ids'][0].numpy()
                combined_index = [l if not pred_mask[i] else predicted_index[i] for i, l in enumerate(original_index)]

                if 'roberta-large' not in model_name:
                    predicted_token = [tokenizer.convert_ids_to_tokens([combined_index[x]])[0] for x in
                                       range(1, length - 1)]
                    generated_text = ''
                    for i, token in enumerate(predicted_token):

                        if token.startswith('##'):
                            generated_text = generated_text + token.replace('##', '')
                        else:
                            if i == 0:
                                generated_text = generated_text + token
                            else:
                                generated_text = generated_text + ' ' + token
                else:
                    combined_index = combined_index[1:-1]
                    generated_text = tokenizer.decode(combined_index)

                if 'roberta-large' not in model_name:
                    generated_text = ' '.join(generated_text.split()[4:]).lower()
                    print(generated_text)

                else:
                    generated_text = ' '.join(generated_text.split()[3:]).lower()
                    print(generated_text)

                f_out.write(generated_text + '\t' + str(sentence_list[sentence_id][-1]) + '\n')

        f_out.close()
