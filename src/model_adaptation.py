"""Custom models for few-shot learning specific operations."""

import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, \
    BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import *
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model, StableDropout, \
    ContextPooler, DebertaV2OnlyMLMHead
from transformers.models.deberta.modeling_deberta import DebertaPreTrainedModel, DebertaModel, StableDropout, \
    ContextPooler, DebertaOnlyMLMHead
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from loss import stable_kl, CeCriterion, KlCriterion, entropy, SymKlCriterion, ContrastiveLoss
from processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, \
    bound_mapping
import logging

logger = logging.getLogger(__name__)

class RobertaAdaptForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaAdaModel(config)
        # self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.hidden_size = config.hidden_size

        # self.map = nn.Linear(config.hidden_size, config.hidden_size)

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.K = 1
        self.step_size = 1e-5
        self.adv_lc = SymKlCriterion()
        self.contra_lc = ContrastiveLoss()

        # self.step_size=config.step_size

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

        sequence_mask_output_1 = self.lm_head.dense(sequence_mask_output_1)
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
        contra_loss = self.contra_lc(sequence_mask_output_1.unsqueeze(1), sequence_mask_output_2.unsqueeze(0),
                                     pair_labels)
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

        blocked_indices = (block_flag == 1).nonzero(as_tuple=False).reshape((bz, self.model_args.prompt_length, 2))[:,
                          :, 1]

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
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos,
                               inputs_embeds=inputs_embeds)

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


class AdapeterLayer(nn.Module):
    def __init__(self, n_in, n_out=None, adapter_dim=16, adapter_choice='lora'):
        super(AdapeterLayer, self).__init__()
        if not n_out:
            n_out = n_in

        self.adapter_choice = adapter_choice

        if self.adapter_choice == 'lora':

            #self.adapter_dim = int(n_in/self.adapter_alpha)
            self.adapter_dim = adapter_dim
            self.adapter_proj_1 = nn.Linear(n_in, adapter_dim, bias=False)
            nn.init.normal_(self.adapter_proj_1.weight, std=0.02)
            self.adapter_proj_2 = nn.Linear(adapter_dim, n_out, bias=False)
            nn.init.normal_(self.adapter_proj_2.weight, std=0.02)
        elif self.adapter_choice == 'linear_after':
            # self.adapter_dim = adapter_dim
            # self.adapter_proj_1 = nn.Linear(n_out, n_out, bias=False)
            # self.adapter_proj_1.weight = torch.nn.Parameter(torch.eye(n_out))

            self.adapter_dim = adapter_dim
            self.adapter_proj_1 = nn.Linear(n_out, adapter_dim, bias=False)
            nn.init.normal_(self.adapter_proj_1.weight, std=0.02)
            self.adapter_proj_2 = nn.Linear(adapter_dim, n_out, bias=False)
            nn.init.normal_(self.adapter_proj_2.weight, std=0.02)
        else:
            # self.adapter_dim = adapter_dim
            # self.adapter_proj_1 = nn.Linear(n_out, n_out, bias=False)
            # self.adapter_proj_1.weight = torch.nn.Parameter(torch.eye(n_out))

            self.adapter_dim = adapter_dim
            self.adapter_proj_1 = nn.Linear(n_out, n_out, bias=False)
            # nn.init.normal_(self.adapter_proj_1.weight, std=0.02)
            # self.adapter_proj_2 = nn.Linear(adapter_dim, n_out, bias=False)
            # nn.init.normal_(self.adapter_proj_2.weight, std=0.02)


            #nn.init.normal_(self.adapter_proj_1.weight, std=0.02)



    def forward(self, x):
        if self.adapter_choice == 'lora':
            result = torch.matmul(x, self.adapter_proj_1.weight.type_as(x).T)
            return torch.matmul(result, self.adapter_proj_2.weight.type_as(x).T)
        elif self.adapter_choice == 'linear_after':
            result = torch.matmul(x, self.adapter_proj_1.weight.type_as(x).T)
            result = torch.matmul(result, self.adapter_proj_2.weight.type_as(x).T)
            return result + x

        else:
            result = torch.matmul(x, self.adapter_proj_1.weight.type_as(x).T)
            return result


class RobertaAdaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.config = config


        self.adaptation_layer = AdapeterLayer(n_in=config.intermediate_size, n_out=config.hidden_size,
                                          adapter_dim=config.adapter_dim, adapter_choice=config.adapter_choice)

        # self.adaptation_layer_skip = AdapeterLayer(n_in=config.intermediate_size, n_out=config.hidden_size,
        #                                       adapter_dim=config.adapter_dim, adapter_choice=config.adapter_choice)



        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        #print(self.dense(hidden_states).abs().mean(), self.adaptation_layer(hidden_states).abs().mean())
        if self.config.adapter_choice == 'lora':
            hidden_states = self.dense(hidden_states) +self.adaptation_layer(hidden_states)
        else:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.adaptation_layer(hidden_states)


        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # hidden_states = self.adaptation_layer_skip(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class RobertaAdaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.adaptation_layer = AdapeterLayer(n_in=config.intermediate_size, n_out=config.hidden_size,
                                              adapter_dim=config.adapter_dim, adapter_choice=config.adapter_choice)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        if self.config.adapter_choice == 'lora':
            hidden_states = self.dense(hidden_states) + self.adaptation_layer(hidden_states)
        else:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.adaptation_layer(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # input_tensor = hidden_states
        # hidden_states = self.adaptation_layer(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta
class RobertaAdaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = RobertaSelfAttention(config)
        self.output = RobertaAdaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta
class RobertaAdaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAdaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaAdaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class RobertaAdaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaAdaLayer(config) for _ in range(config.num_hidden_layers)])
        self.skip = 2

    def learn_init(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True):





            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None
            all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

            next_decoder_cache = () if use_cache else None
            self.skip_list = []
            for i, layer_module in enumerate(self.layer):
                # if i+1 % self.skip

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None

                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    if use_cache:
                        logger.warning(
                            "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                            "`use_cache=False`..."
                        )
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, past_key_value, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )

                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        next_decoder_cache,
                        all_hidden_states,
                        all_self_attentions,
                        all_cross_attentions,
                    ]
                    if v is not None
                )
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            # if (i+1) % 3 == 0:
            #    continue
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
class RobertaAdaModel(RobertaPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need`_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaAdaEncoder(config)


        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        #self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
