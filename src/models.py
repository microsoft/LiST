"""Custom models for few-shot learning specific operations."""

import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model, StableDropout, ContextPooler, DebertaV2OnlyMLMHead
from transformers.models.deberta.modeling_deberta import DebertaPreTrainedModel, DebertaModel, StableDropout, ContextPooler, DebertaOnlyMLMHead
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from loss import stable_kl, CeCriterion, KlCriterion, entropy, SymKlCriterion, ContrastiveLoss
from processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping
import logging
from model_adaptation import RobertaAdaModel, BertAdaModel
import os

logger = logging.getLogger(__name__)

def generate_noise(embed, mask, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) *  epsilon
    noise.detach()
    noise.requires_grad_()
    return noise

def norm_grad(grad, eff_grad=None, sentence_level=False, norm_p='max', epsilon=1e-5):
    if norm_p == 'l2':
        if sentence_level:
            direction = grad / (torch.norm(grad, dim=(-2, -1), keepdim=True) + epsilon)
        else:
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + epsilon)
    elif norm_p == 'l1':
        direction = grad.sign()
    else:
        if sentence_level:
            direction = grad / (grad.abs().max((-2, -1), keepdim=True)[0] + epsilon)
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + epsilon)
            eff_direction = eff_grad / (grad.abs().max(-1, keepdim=True)[0] + epsilon)
    return direction, eff_direction



def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


class LMForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config, model_args, data_args):
        super().__init__(config)
        self.model_args = model_args
        self.data_args = data_args
        self.config = config


        # Create config
        num_labels = num_labels_mapping[data_args.task_name]
        self.num_labels = num_labels
        config.adapter_dim = model_args.adapter_dim
        config.adapter_alpha = model_args.adapter_alpha
        config.adapter_choice = model_args.adapter_choice
        self.config = config

        if 'prompt' in model_args.few_shot_type:
            if config.model_type == 'roberta':
                model_fn = RobertaForPromptFinetuning
            elif config.model_type == 'bert':
                model_fn = BertForPromptFinetuning
            elif config.model_type == 'deberta':
                model_fn = DebertaForPromptFinetuning
            elif config.model_type == 'deberta-v2':
                model_fn = Debertav2ForPromptFinetuning
            elif config.model_type == 't5':
                self.lm_model = T5ForPromptFinetuning(config)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        if config.model_type == 't5':
            self.lm_model.T5 =  self.lm_model.T5.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

        else:

            self.lm_model = model_fn.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )



        # Pass dataset and argument information to the model
        if data_args.prompt:
            self.lm_model.label_word_list = torch.tensor(data_args.label_word_list).long().cuda()
        if output_modes_mapping[data_args.task_name] == 'regression':
            # lower / upper bounds
            self.lm_model.lb, self.lm_model.ub = bound_mapping[data_args.task_name]
        self.lm_model.model_args = model_args
        self.lm_model.data_args = data_args
        self.hidden_size = config.hidden_size

        if self.data_args.continuous_prompt == 1:
            self.prompt_embeddings = torch.nn.Embedding(self.data_args.prompt_length, self.hidden_size)
        else:
            self.prompt_embeddings = None


        self.prompt_embeddings = torch.nn.Embedding(self.data_args.prompt_length, self.hidden_size)

        if self.model_args.adapter_choice != 'none':
            self.init_adapter(std=self.model_args.adapter_init_std)


        self.prompt_encoder = None
        if self.data_args.continuous_prompt == 1:
            self.init_embedding()

    def init_adapter(self, std):
        with torch.no_grad():
            for name, param in self.lm_model.named_parameters():
                init_value = 0
                if 'adapter_proj' in name:

                    if self.model_args.adapter_choice == 'simple':

                        init_value = torch.eye(param.size(0))

                    if std > 0:

                        init_value += torch.normal(0, std, size=param.size())
                    param.copy_(init_value)




    def freeze_lm(self):
        for name, param in self.lm_model.named_parameters():
            param.requires_grad = False

    def freeze_lm_encoder(self):
        for name, param in self.lm_model.named_parameters():
            if 'lm_head' in name or ('cls' in name):
                print(name)
                continue
            param.requires_grad = False

    def freeze_lm_finetune_bias(self):
        for name, param in self.lm_model.named_parameters():
            if "bias" in name:
                print(name)
                continue
            param.requires_grad = False

    def freeze_lm_component(self, component):

        if 'attention' in component:
            for name, param in self.lm_model.named_parameters():
                if 'attention' in name:
                    if 'output' in component:
                        if 'output' in name:
                            continue
                    else:
                        continue
                param.requires_grad = False
            self.unfreeze_classification_head()
        elif 'feedforward' in component:
            for name, param in self.lm_model.named_parameters():
                if 'dense' in name and 'attention' not in name:
                    if 'output' in component:
                        if 'output' in name:
                            continue
                    else:
                        if 'intermediate' in component:
                            if 'intermediate' in name:
                                continue
                param.requires_grad = False
            self.unfreeze_classification_head()
        elif component == 'adapter':
            for name, param in self.lm_model.named_parameters():
                if 'adapter' in name:
                    continue

                param.requires_grad = False
            self.unfreeze_classification_head()
        elif 'embedding' in component:
            for name, param in self.lm_model.named_parameters():
                if 'embedding' in name:
                    continue


                param.requires_grad = False
            self.unfreeze_classification_head()
        elif 'bias' in component:
            for name, param in self.lm_model.named_parameters():
                if 'bias' in name:
                    continue
                param.requires_grad = False
            self.unfreeze_classification_head()
        elif 'head' in component:
            for name, param in self.lm_model.named_parameters():
                param.requires_grad = False
            self.unfreeze_classification_head()


        elif "prompt_emb" in component:

            for name, param in self.lm_model.named_parameters():
                if 'prompt_emb' in name:
                    continue
                param.requires_grad = False





    def unfreeze_classification_head(self):
        for name, param in self.lm_model.named_parameters():
            if 'lm_head' in name or ('cls' in name) or ('classifier' in name):
                param.requires_grad = True



    def freeeze_lm_k_layers(self, k):

        keep_layers = []
        update_parameters = []
        for i in range(k):
            keep_layers.append('layer.'+str(23-i))

        for name, param in self.lm_model.named_parameters():
            update = False
            for layer_num in keep_layers:
                if layer_num in name:
                    if 'dense' in name and 'attention' not in name:
                        if 'output' in name:
                            print(name)
                            update_parameters.append(name)
                            update = True

            if not update:
                param.requires_grad = False
        self.unfreeze_classification_head()


    def unfreeze_lm(self):
        for param in self.lm_model.parameters():
            param.requires_grad = True


    def init_embedding(self):

        rand_id = torch.randint(100, self.config.vocab_size, (self.data_args.prompt_length,)).long()
        rand_emb = self.lm_model.embed_encode(rand_id)
        self.prompt_embeddings = self.prompt_embeddings.from_pretrained(rand_emb, freeze=False)



    def get_adv_loss(self,
                     input_ids=None,
                     attention_mask=None,
                     mask_pos=None,
                     labels=None,
                     inputs_embeds=None):

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

        input_args = [input_ids, attention_mask, mask_pos, labels, None, 1]
        embed = self.forward(*input_args)
        noise = generate_noise(embed, attention_mask)

        for step in range(0, self.K):
            vat_args = [input_ids, attention_mask, mask_pos, labels, embed + noise, 2]
            adv_logits, _ = self.forward(*vat_args)
            adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
            try:
                delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
            except:
                import pdb
                pdb.set_trace()

            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad, eff_noise = norm_grad(delta_grad, eff_grad=eff_delta_grad)
            noise = noise + delta_grad * self.step_size
            # noise, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()

        vat_args = [input_ids, attention_mask, mask_pos, labels, embed + noise, 2]

        adv_logits, sequence_mask_output = self.forward(*vat_args)
        # ori_args = model(*ori_args)
        # aug_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 2, (embed + native_noise).detach()]

        adv_loss = self.adv_lc(adv_logits, logits, reduction='none')
        return adv_loss

    def embed_encode(self, input_ids):
        embedding_output = self.lm_model.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        # import pdb
        # pdb.set_trace()
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        if inputs_embeds is None:
            outputs = self.lm_model(
                input_ids,
                attention_mask=attention_mask
            )
        else:
            outputs = self.lm_model(
                None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # sequence_mask_output = self.lm_head.dense(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:

            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        if self.model_args.hybrid == 1:
            cls_logits = self.classifier(sequence_output)
            return (logits, cls_logits), sequence_mask_output

        return logits, sequence_mask_output

    def generate_continuous_prompt_inputs(self, input_ids, block_flag):


        inputs_embeds = self.lm_model.embed_encode(input_ids)
        bz = inputs_embeds.shape[0]

        try:
            replace_embeds = self.prompt_embeddings(
                torch.LongTensor(list(range(self.data_args.prompt_length))).to(inputs_embeds.device))
        except:
            import pdb
            pdb.set_trace()
            replace_embeds = self.prompt_embeddings(
                torch.LongTensor(list(range(self.data_args.prompt_length))))

        replace_embeds = replace_embeds.unsqueeze(0)  # [batch_size, prompt_length, embed_size]


        if self.prompt_encoder is not None:
            replace_embeds = self.prompt_encoder(replace_embeds)

        blocked_indices = (block_flag == 1).nonzero(as_tuple=False).reshape((bz, self.data_args.prompt_length, 2))[:, :, 1]

        for bidx in range(bz):
            for i in range(blocked_indices.shape[1]):
                inputs_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[:, i, :].squeeze()

        return inputs_embeds

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            mask_pos=None,
            labels=None,
            inputs_embeds=None,
            fwd_type=0,
            block_flag=None,
            *args,
            **kwargs
    ):


        if 't5' in self.config.model_type:
            logits, sequence_mask_output = self.lm_model.encode(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        else:

            if fwd_type == 2:
                assert inputs_embeds is not None
                if token_type_ids is not None:
                    return self.lm_model.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mask_pos=mask_pos,
                                                inputs_embeds=inputs_embeds)
                else:
                    return self.lm_model.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos,
                                                inputs_embeds=inputs_embeds)

            elif fwd_type == 1:
                return self.lm_model.embed_encode(input_ids)



            if self.data_args.continuous_prompt == 1 and block_flag is not None and block_flag[0] is not None:
                inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

            if fwd_type == 3:
                if token_type_ids is not None:
                    prediction_mask_scores = self.lm_model.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mask_pos=mask_pos,
                                                inputs_embeds=inputs_embeds, return_full_softmax=True)
                else:
                    prediction_mask_scores = self.lm_model.encode(input_ids, attention_mask, mask_pos, inputs_embeds, return_full_softmax=True)
                if labels is not None:
                    return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
                return prediction_mask_scores

            if token_type_ids is not None:
                logits, sequence_mask_output = self.lm_model.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mask_pos=mask_pos,
                                                inputs_embeds=inputs_embeds)
            else:
                logits, sequence_mask_output = self.lm_model.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

            if fwd_type == 4:
                return logits, sequence_mask_output

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                                      (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:


                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction='batchmean')
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        return ((loss,) + output) if loss is not None else output


    def from_pretrained(self, pretrained_model_name_or_path, *model_args, **kwargs):

        self.lm_model = self.lm_model.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        if self.data_args.prompt:
            self.lm_model.label_word_list = torch.tensor(self.data_args.label_word_list).long().cuda()
        if output_modes_mapping[self.data_args.task_name] == 'regression':
            # lower / upper bounds
            self.lm_model.lb, self.lm_model.ub = bound_mapping[self.data_args.task_name]
        self.lm_model.model_args = self.model_args
        self.lm_model.data_args = self.data_args

        return self

    def load_model(self, checkpoint):

        if os.path.isfile(checkpoint):
            model_state_dict = torch.load(checkpoint)
            self.load_state_dict(model_state_dict, strict=False)



class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class T5ForPromptFinetuning(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)


        self.T5 = T5ForConditionalGeneration(config)

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None


    def get_labels(self, input_ids):
        batch_size = input_ids.size(0)
        # new_labels = torch.tensor([3,32099,1] * batch_size).to(labels.device)
        # prefix = torch.tensor([32099] * batch_size).to(labels.device)
        # ending = torch.tensor([1] * batch_size).to(labels.device)
        # prefix_labels = torch.cat((start.unsqueeze(1), prefix.unsqueeze(1)), 1)
        # prefix_labels = torch.cat(( prefix_labels, labels.unsqueeze(1)), 1)
        new_labels =  torch.tensor([3, 32099,1]).to(input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        return new_labels





    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None,
               return_full_softmax=False, labels=None):


        if labels is not None:
            t5_labels = self.get_labels(input_ids)
            outputs = self.T5(input_ids=input_ids, attention_mask=attention_mask, labels=t5_labels)

            prediction_mask_scores = outputs.logits[:, 2, :]

            logits = []

            for label_id in range(len(self.label_word_list)):
                logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
            logits = torch.cat(logits, -1)

        return logits, prediction_mask_scores



class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        if config.adapter_choice != 'none':
            self.bert = BertAdaModel(config)
        else:
            self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def embed_encode(self, input_ids):
        embedding_output = self.bert.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        batch_size = input_ids.size(0)

        
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        if inputs_embeds is None:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.bert(
                None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds
            )


        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]



        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        if self.model_args.hybrid == 1:
            cls_logits = self.classifier(sequence_output)
            return (logits, cls_logits), sequence_mask_output


        return logits, sequence_mask_output


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        fwd_type=0,
        block_flag=None
    ):
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)

        if self.data_args.continuous_prompt == 1 and block_flag is not None:
            inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

        if self.model_args.hybrid == 1:

            logits = logits[0]
            cls_logits = logits[1]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction='batchmean')
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        return ((loss,) + output) if loss is not None else output







class RobertaForPromptFinetuning(BertPreTrainedModel):


    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        if config.adapter_choice != 'none':
            self.roberta = RobertaAdaModel(config)
        else:
            self.roberta = RobertaModel(config)
        # self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.hidden_size = config.hidden_size

        # self.map = nn.Linear(config.hidden_size, config.hidden_size)


        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.K = 1
        self.step_size=1e-5
        self.adv_lc = SymKlCriterion()
        self.contra_lc = ContrastiveLoss()

        #self.step_size=config.step_size

        # For regression
        self.lb = None
        self.ub = None

        self.tokenizer = None

        self.prompt_embeddings = None
        self.lstm_head = None
        self.mlp_head = None
        self.mlp = None

        # For auto label search.
        self.return_full_softmax = None

        #self.init_weights()
        # else:
        #     raise ValueError('unknown prompt_encoder_type.')




    def get_constrast_loss(self,
                    input_ids=None,
                    attention_mask=None,
                    mask_pos=None,
                    labels=None,
                    inputs_embeds=None,
                    block_flag=None):

        self.cos = nn.CosineSimilarity(dim=-1)

        if self.data_args.continuous_prompt == 1:
            inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)



        _, sequence_mask_output_1 = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)
        _, sequence_mask_output_2 = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

        sequence_mask_output_1= self.lm_head.dense(sequence_mask_output_1)
        sequence_mask_output_2 = self.lm_head.dense(sequence_mask_output_2)
        # input_args = [input_ids, attention_mask, mask_pos, labels, None, 1]
        # embed = self.forward(*input_args)
        #
        # vat_args = [input_ids, attention_mask, mask_pos, labels, embed, 2]
        #
        # adv_logits, outputs = self.forward(*vat_args)
        #
        # logit_mask = F.softmax(logits, dim=-1)[torch.arange(adv_logits.size(0)), labels] > 0.7
        #
        # outputs = outputs[logit_mask]
        # seq_outputs = sequence_mask_output[logit_mask]
        # new_label = labels[logit_mask]
        # #
        # #
        # rand_perm = torch.randperm(outputs.size(0))
        # rand_outputs = outputs[rand_perm, :]
        # rand_label = new_label[rand_perm]
        # pair_label = (new_label == rand_label).long()
        #
        # seq_outputs = self.map(seq_outputs)
        # rand_outputs = self.map(rand_outputs)

        pair_labels = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        contra_loss = self.contra_lc(sequence_mask_output_1.unsqueeze(1), sequence_mask_output_2.unsqueeze(0), pair_labels)
        if torch.isnan(contra_loss):
            return 0

        return contra_loss




    def get_adv_loss(self,
                    input_ids=None,
                    attention_mask=None,
                    mask_pos=None,
                    labels=None,
                    inputs_embeds=None):

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

        input_args = [input_ids, attention_mask, mask_pos, labels, None, 1]
        embed = self.forward(*input_args)
        noise = generate_noise(embed, attention_mask)


        for step in range(0, self.K):
            vat_args = [input_ids, attention_mask, mask_pos, labels, embed + noise, 2]
            adv_logits, _ = self.forward(*vat_args)
            adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
            try:
                delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
            except:
                import pdb
                pdb.set_trace()

            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad, eff_noise = norm_grad(delta_grad, eff_grad=eff_delta_grad)
            noise = noise + delta_grad * self.step_size
            # noise, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()


        vat_args = [input_ids, attention_mask, mask_pos, labels, embed + noise, 2]

        adv_logits, sequence_mask_output = self.forward(*vat_args)
        # ori_args = model(*ori_args)
        # aug_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 2, (embed + native_noise).detach()]

        adv_loss = self.adv_lc(adv_logits, logits, reduction='none')
        return adv_loss

    def embed_encode(self, input_ids):
        embedding_output = self.roberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        if inputs_embeds is None:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask
            )
        else:
            outputs = self.roberta(
                None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            )


        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]



        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        if self.model_args.hybrid == 1:
            cls_logits = self.classifier(sequence_output)
            return (logits, cls_logits), sequence_mask_output


        return logits, sequence_mask_output

    def generate_continuous_prompt_inputs(self, input_ids, block_flag):

        inputs_embeds = self.embed_encode(input_ids)
        bz = inputs_embeds.shape[0]

        try:
            replace_embeds = self.prompt_embeddings(
                torch.LongTensor(list(range(1))).to(inputs_embeds.device))
        except:
            import pdb
            pdb.set_trace()
            replace_embeds = self.prompt_embeddings(
                torch.LongTensor(list(range(1))))

        replace_embeds = replace_embeds.unsqueeze(0)  # [batch_size, prompt_length, embed_size]

        # if self.model_args.prompt_encoder_type == "lstm":
        #     replace_embeds = self.lstm_head(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
        #     if self.prompt_length == 1:
        #         replace_embeds = self.mlp_head(replace_embeds)
        #     else:
        #         replace_embeds = self.mlp_head(replace_embeds).squeeze()

        # elif self.model_args.prompt_encoder_type == "mlp":
        replace_embeds = self.mlp(replace_embeds)
        # else:
        #     raise ValueError("unknown prompt_encoder_type.")

        blocked_indices = (block_flag == 1).nonzero(as_tuple=False).reshape((bz, self.model_args.prompt_length, 2))[:, :, 1]

        for bidx in range(bz):
            for i in range(blocked_indices.shape[1]):
                inputs_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

        return inputs_embeds



    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        fwd_type=0,
        block_flag=None
    ):
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)

        if self.data_args.continuous_prompt == 1 and block_flag is not None:
            inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

        if self.model_args.hybrid == 1:

            logits = logits[0]
            cls_logits = logits[1]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction='batchmean')
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        return ((loss,) + output) if loss is not None else output


class RobertaForSequenceClassification(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        if config.adapter_choice != 'none':
            self.roberta = RobertaAdaModel(config, add_pooling_layer=False)
        else:
            self.roberta = RobertaModel(config, add_pooling_layer=False)

        #self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def freeze_lm_component(self, component):

        if 'attention' in component:
            for name, param in self.roberta.named_parameters():
                if 'attention' in name:
                    if 'output' in component:
                        if 'output' in name:
                            continue
                    else:
                        continue
                param.requires_grad = False
        elif 'feedforward' in component:
            for name, param in self.roberta.named_parameters():
                if 'dense' in name and 'attention' not in name:
                    if 'output' in component:
                        if 'output' in name:
                            continue
                    else:
                        if 'intermediate' in component:
                            if 'intermediate' in name:
                                continue
                param.requires_grad = False
        elif component == 'adapter':
            for name, param in self.roberta.named_parameters():
                if 'adapter' in name:
                    continue

                param.requires_grad = False
        elif 'embedding' in component:
            for name, param in self.roberta.named_parameters():
                if 'embedding' in name:
                    continue
                    # if 'lm_head' in name:
                    #
                    # if 'output' in name:
                    #     continue

                param.requires_grad = False
        elif 'bias' in component:
            for name, param in self.roberta.named_parameters():
                if 'bias' in name:
                    continue
                    # if 'lm_head' in name:
                    #
                    # if 'output' in name:
                    #     continue

                param.requires_grad = False
        elif 'head' in component:
            for name, param in self.roberta.named_parameters():
                param.requires_grad = False



        self.unfreeze_classification_head()

    def unfreeze_classification_head(self):
        for name, param in self.roberta.named_parameters():
            if 'lm_head' in name or ('cls' in name) or ('classifier' in name):
                param.requires_grad = True


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))



        #output = (logits,) + outputs[2:]
        output = (logits,)


        return ((loss,) + output) if loss is not None else output



class DebertaForPromptFinetuning(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        #self.deberta = DebertaV2Model(config)

        self.deberta = DebertaModel(config)
        self.cls = DebertaOnlyMLMHead(config)

        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = torch.nn.Linear(output_dim, self.num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.dropout = StableDropout(drop_out)

        classification_list = [self.pooler, self.dropout,self.classifier]

        self.classifier = nn.Sequential(*classification_list)
        # self.cls = DebertaV2OnlyMLMHead(config)

        self.map = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.K = 1
        self.step_size=1e-5
        self.adv_lc = SymKlCriterion()
        self.contra_lc = ContrastiveLoss()
        # import pdb
        # pdb.set_trace()
        #self.step_size=config.step_size

        # For regression
        self.lb = None
        self.ub = None


        # For auto label search.
        self.return_full_softmax = None

    def get_constrast_loss(self,
                    input_ids=None,
                    attention_mask=None,
                    mask_pos=None,
                    labels=None,
                    inputs_embeds=None):

        self.cos = nn.CosineSimilarity(dim=-1)


        _, sequence_mask_output_1 = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)
        _, sequence_mask_output_2 = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

        sequence_mask_output_1= self.lm_head.dense(sequence_mask_output_1)
        sequence_mask_output_2 = self.lm_head.dense(sequence_mask_output_2)
        # input_args = [input_ids, attention_mask, mask_pos, labels, None, 1]
        # embed = self.forward(*input_args)
        #
        # vat_args = [input_ids, attention_mask, mask_pos, labels, embed, 2]
        #
        # adv_logits, outputs = self.forward(*vat_args)
        #
        # logit_mask = F.softmax(logits, dim=-1)[torch.arange(adv_logits.size(0)), labels] > 0.7
        #
        # outputs = outputs[logit_mask]
        # seq_outputs = sequence_mask_output[logit_mask]
        # new_label = labels[logit_mask]
        # #
        # #
        # rand_perm = torch.randperm(outputs.size(0))
        # rand_outputs = outputs[rand_perm, :]
        # rand_label = new_label[rand_perm]
        # pair_label = (new_label == rand_label).long()
        #
        # seq_outputs = self.map(seq_outputs)
        # rand_outputs = self.map(rand_outputs)

        pair_labels = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # import  pdb
        # pdb.set_trace()


        contra_loss = self.contra_lc(sequence_mask_output_1.unsqueeze(1), sequence_mask_output_2.unsqueeze(0), pair_labels)



        if torch.isnan(contra_loss):
            return 0

        return contra_loss





    def get_adv_loss(self,
                    input_ids=None,
                    attention_mask=None,
                    mask_pos=None,
                    labels=None,
                    inputs_embeds=None):

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

        input_args = [input_ids, attention_mask, mask_pos, labels, None, 1]
        embed = self.forward(*input_args)
        noise = generate_noise(embed, attention_mask)


        for step in range(0, self.K):
            vat_args = [input_ids, attention_mask, mask_pos, labels, embed + noise, 2]
            adv_logits, _ = self.forward(*vat_args)
            adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
            try:
                delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
            except:
                import pdb
                pdb.set_trace()

            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad, eff_noise = norm_grad(delta_grad, eff_grad=eff_delta_grad)
            noise = noise + delta_grad * self.step_size
            # noise, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()


        vat_args = [input_ids, attention_mask, mask_pos, labels, embed + noise, 2]

        adv_logits, sequence_mask_output = self.forward(*vat_args)
        # ori_args = model(*ori_args)
        # aug_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 2, (embed + native_noise).detach()]

        adv_loss = self.adv_lc(adv_logits, logits)
        return adv_loss

    def embed_encode(self, input_ids):
        embedding_output = self.deberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None,
               return_full_softmax=False):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()


        # Encode everything
        if inputs_embeds is None:
            outputs = self.deberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.deberta(
                None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds
            )

        # Get <mask> token representation
        sequence_output = outputs[0]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # sequence_mask_output = self.lm_head.dense(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        if self.model_args.hybrid == 1:
            cls_logits = self.classifier(sequence_output)
            return (logits, cls_logits), sequence_mask_output

        return logits, sequence_mask_output

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            mask_pos=None,
            labels=None,
            inputs_embeds=None,
            fwd_type=0,
            block_flag=None
    ):
        
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)



        if self.data_args.continuous_prompt == 1 and block_flag is not None:
            inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

        if self.model_args.hybrid == 1:
            logits = logits[0]
            cls_logits = logits[1]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                                      (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction='batchmean')
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        return ((loss,) + output) if loss is not None else output



class Debertav2ForPromptFinetuning(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)
        self.cls = DebertaV2OnlyMLMHead(config)

        #self.deberta = DebertaModel(config)
        #self.cls = DebertaOnlyMLMHead(config)

        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = torch.nn.Linear(output_dim, self.num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.dropout = StableDropout(drop_out)

        classification_list = [self.pooler, self.dropout,self.classifier]

        self.classifier = nn.Sequential(*classification_list)
        # self.cls = DebertaV2OnlyMLMHead(config)

        self.map = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.K = 1
        self.step_size=1e-5
        self.adv_lc = SymKlCriterion()
        self.contra_lc = ContrastiveLoss()
        # import pdb
        # pdb.set_trace()
        #self.step_size=config.step_size

        # For regression
        self.lb = None
        self.ub = None


        # For auto label search.
        self.return_full_softmax = None

    def get_constrast_loss(self,
                    input_ids=None,
                    attention_mask=None,
                    mask_pos=None,
                    labels=None,
                    inputs_embeds=None):

        self.cos = nn.CosineSimilarity(dim=-1)


        _, sequence_mask_output_1 = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)
        _, sequence_mask_output_2 = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

        sequence_mask_output_1= self.lm_head.dense(sequence_mask_output_1)
        sequence_mask_output_2 = self.lm_head.dense(sequence_mask_output_2)
        # input_args = [input_ids, attention_mask, mask_pos, labels, None, 1]
        # embed = self.forward(*input_args)
        #
        # vat_args = [input_ids, attention_mask, mask_pos, labels, embed, 2]
        #
        # adv_logits, outputs = self.forward(*vat_args)
        #
        # logit_mask = F.softmax(logits, dim=-1)[torch.arange(adv_logits.size(0)), labels] > 0.7
        #
        # outputs = outputs[logit_mask]
        # seq_outputs = sequence_mask_output[logit_mask]
        # new_label = labels[logit_mask]
        # #
        # #
        # rand_perm = torch.randperm(outputs.size(0))
        # rand_outputs = outputs[rand_perm, :]
        # rand_label = new_label[rand_perm]
        # pair_label = (new_label == rand_label).long()
        #
        # seq_outputs = self.map(seq_outputs)
        # rand_outputs = self.map(rand_outputs)

        pair_labels = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # import  pdb
        # pdb.set_trace()


        contra_loss = self.contra_lc(sequence_mask_output_1.unsqueeze(1), sequence_mask_output_2.unsqueeze(0), pair_labels)



        if torch.isnan(contra_loss):
            return 0

        return contra_loss





    def get_adv_loss(self,
                    input_ids=None,
                    attention_mask=None,
                    mask_pos=None,
                    labels=None,
                    inputs_embeds=None):

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

        input_args = [input_ids, attention_mask, mask_pos, labels, None, 1]
        embed = self.forward(*input_args)
        noise = generate_noise(embed, attention_mask)


        for step in range(0, self.K):
            vat_args = [input_ids, attention_mask, mask_pos, labels, embed + noise, 2]
            adv_logits, _ = self.forward(*vat_args)
            adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
            try:
                delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
            except:
                import pdb
                pdb.set_trace()

            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad, eff_noise = norm_grad(delta_grad, eff_grad=eff_delta_grad)
            noise = noise + delta_grad * self.step_size
            # noise, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()


        vat_args = [input_ids, attention_mask, mask_pos, labels, embed + noise, 2]

        adv_logits, sequence_mask_output = self.forward(*vat_args)
        # ori_args = model(*ori_args)
        # aug_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 2, (embed + native_noise).detach()]

        adv_loss = self.adv_lc(adv_logits, logits)
        return adv_loss

    def embed_encode(self, input_ids):
        embedding_output = self.deberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        if inputs_embeds is None:
            outputs =  self.deberta(
                input_ids,
                attention_mask=attention_mask
            )
        else:
            outputs =  self.deberta(
                None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            )


        # Get <mask> token representation
        sequence_output = outputs[0]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]


        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        return logits, sequence_mask_output


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        fwd_type=0,
        block_flag=None
    ):
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

        loss = None


        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction='batchmean')
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                    if self.model_args.hybrid == 1:
                        cls_loss = loss_fct(cls_logits.view(-1, cls_logits.size(-1)), labels.view(-1))
                        loss = loss + cls_loss

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        return ((loss,) + output) if loss is not None else output

