# Copyright (c) Microsoft. All rights reserved.
from copy import deepcopy
import torch
import logging
import random
from torch.nn import Parameter, ParameterDict
import torch.nn.functional as F
from loss import stable_kl, CeCriterion, KlCriterion, entropy
from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM
import torch
from collections import OrderedDict
logger = logging.getLogger(__name__)

def generate_noise(embed, mask, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) *  epsilon
    noise.detach()
    noise.requires_grad_()
    return noise

#
class SmartPerturbation():
    def __init__(self,
                 epsilon=1e-6,
                 multi_gpu_on=False,
                 step_size=1e-2,
                 noise_var=1e-5,
                 norm_p='inf',
                 k=1,
                 fp16=False,
                 encoder_type=EncoderModelType.BERT,
                 loss_map=[],
                 norm_level=0,
                 num_train_step=0):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta
        self.step_size = step_size
        self.multi_gpu_on = multi_gpu_on
        self.fp16 = fp16
        self.K = k
        self.count = 0
        # self.modelLM = RobertaForMaskedLM.from_pretrained('roberta-large')
        # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p
        self.encoder_type = encoder_type
        self.loss_map = loss_map
        self.norm_level = norm_level > 0
        assert len(loss_map) > 0


    def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
        if self.norm_p == 'l2':
            if sentence_level:
                direction = grad / (torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon)
            else:
                direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + self.epsilon)
        elif self.norm_p == 'l1':
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (grad.abs().max((-2, -1), keepdim=True)[0] + self.epsilon)
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
                eff_direction = eff_grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
        return direction, eff_direction

    def forward(self,
                model,
                input_ids=None,
                attention_mask=None,
                mask_pos=None,
                labels=None):

        embed = model(*vat_args)
        noise = generate_noise(embed, attention_mask, epsilon=self.noise_var)

        for step in range(0, self.K):
            vat_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 2, embed + noise]
            adv_logits = model(*vat_args)
            if task_type == TaskType.Regression:
                adv_loss = F.mse_loss(adv_logits, logits.detach(), reduction='sum')
            else:
                if task_type == TaskType.Ranking:
                    adv_logits = adv_logits.view(-1, pairwise)
                adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)

            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise + delta_grad * self.step_size
            #noise, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()
        self.count += 1

        vat_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 2, (embed + noise)]

        adv_logits = model(*vat_args)
        # ori_args = model(*ori_args)
        #aug_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 2, (embed + native_noise).detach()]

        if task_type == TaskType.Ranking:
            adv_logits = adv_logits.view(-1, pairwise)
        adv_lc = self.loss_map[task_id]
        #mask = torch.max(F.softmax(adv_logits, dim=-1), dim=-1)[0] > 0.7

        if only_logits:
            return torch.tensor([0]).to(logits.device), torch.tensor([0]), torch.tensor([0]), adv_logits

        adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        # vat_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 3, (embed + noise)]
        # vat_output = model(*vat_args)
        # ori_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 2, (embed)]
        # output = model(*ori_args)
        if torch.isnan(adv_loss):
            return torch.tensor([0]).to(logits.device), torch.tensor([0]), torch.tensor([0]), adv_logits
        #adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        #adv_loss = stable_kl(adv_logits, logits.detach())
        return adv_loss, embed.detach().abs().mean(), noise.detach().abs().mean(), adv_logits


class WeightPerturbation():
    def __init__(self,
                 epsilon=1e-6,
                 multi_gpu_on=False,
                 step_size=1e-3,
                 noise_var=1e-5,
                 norm_p='inf',
                 k=1,
                 fp16=False,
                 encoder_type=EncoderModelType.BERT,
                 loss_map=[],
                 norm_level=0,
                 num_train_step=0):
        super(WeightPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta
        self.step_size = step_size
        self.multi_gpu_on = multi_gpu_on
        self.fp16 = fp16
        self.K = k
        self.count = 0
        self.num_train_step = num_train_step
        self.prev_direct={}

        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p
        self.encoder_type = encoder_type
        self.loss_map = loss_map
        self.norm_level = norm_level > 0
        assert len(loss_map) > 0



    def _norm_grad(self, grad):
        if self.norm_p == 'l2':
            direction = grad / (torch.norm(grad) + self.epsilon)
        elif self.norm_p == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max() + self.epsilon)
        return direction


    def _add_noise(self, model):
        for param_key, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data + generate_noise(param, None, epsilon=self.noise_var).detach()

    def forward(self, model,
                logits,
                input_ids,
                token_type_ids,
                attention_mask,
                premise_mask=None,
                hyp_mask=None,
                task_id=0,
                task_type=TaskType.Classification,
                pairwise=1,
                label=None):
        # adv training
        assert task_type in set([TaskType.Classification, TaskType.Ranking, TaskType.Regression]), 'Donot support {} yet'.format(task_type)


        vat_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id]
        f_param = {}
        for param_key, param in model.named_parameters():
            if param.requires_grad:
                f_param[param_key] = param.data.clone()
        adv_logits = model(*vat_args)


        adv_loss = entropy(adv_logits, reduce=False)
        grads = torch.autograd.grad(adv_loss, model.parameters(), only_inputs=True, retain_graph=False)


        f_params_new = self.update_params_sgd(model, grads)

        for param_key, param in model.named_parameters():
            if param.requires_grad:
                param.data = f_params_new[param_key].data  # use data only as f_params_new has graph

        del f_params_new

        adv_logits = model(*vat_args)

        for param_key, param in model.named_parameters():
            if param.requires_grad:
                param.data = f_param[param_key]
        del f_param

        if task_type == TaskType.Ranking:
            adv_logits = adv_logits.view(-1, pairwise)
        adv_lc = self.loss_map[task_id]

        adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)

        if torch.isnan(adv_loss):
            return torch.tensor([0]).to(logits.device), torch.tensor([0]), torch.tensor([0]), adv_logits
        grads = None

        return adv_loss, torch.tensor([0]), torch.tensor([0]), adv_logits


    def update_params_sgd(self, model, grads):
        # supports SGD-like optimizers
        ans = {}
        for i, (name, para) in enumerate(model.named_parameters()):
            if para.requires_grad:
                grad_direct = self._norm_grad(grads[i].detach())
                # if name in self.prev_direct:
                #     new_grad_direct = 0.8 * self.prev_direct[name] + 0.2 * grad_direct
                # else:
                #     new_grad_direct =  grad_direct
                # self.prev_direct[name] = new_grad_direct
                ans[name] = para +  grad_direct * self.step_size  * para.detach().abs().max()
        self.count += 1
        return ans


class WeightADVPerturbation():
    def __init__(self,
                 epsilon=1e-6,
                 multi_gpu_on=False,
                 step_size=1e-2,
                 noise_var=1e-5,
                 norm_p='inf',
                 k=1,
                 fp16=False,
                 encoder_type=EncoderModelType.BERT,
                 loss_map=[],
                 norm_level=0,
                 label=None):
        super(WeightADVPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta
        self.step_size = step_size
        self.multi_gpu_on = multi_gpu_on
        self.fp16 = fp16
        self.K = k
        self.count = 0
        self.adv = True

        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p
        self.encoder_type = encoder_type
        self.loss_map = loss_map
        self.norm_level = norm_level > 0
        assert len(loss_map) > 0



    def _norm_grad(self, grad):

        if self.norm_p == 'l2':
            direction = grad / (torch.norm(grad) + self.epsilon)
        elif self.norm_p == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max() + self.epsilon)
        return direction


    def _add_noise(self, model):
        for param_key, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data + generate_noise(param, None, epsilon=self.noise_var).detach()


    def forward(self, model,
                logits,
                input_ids,
                token_type_ids,
                attention_mask,
                premise_mask=None,
                hyp_mask=None,
                task_id=0,
                task_type=TaskType.Classification,
                pairwise=1,
                label=None):
        # adv training
        assert task_type in set([TaskType.Classification, TaskType.Ranking, TaskType.Regression]), 'Donot support {} yet'.format(task_type)


        vat_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id]
        f_param = {}
        for param_key, param in model.named_parameters():
            if param.requires_grad:
                f_param[param_key] = param.data.clone()


        adv_logits = model(*vat_args)
        task_loss = CeCriterion()

        adv_loss = task_loss(adv_logits, label, ignore_index=-1)

        #adv_loss = entropy(adv_logits, reduce=False)
        grads = torch.autograd.grad(adv_loss, model.parameters(), only_inputs=True, retain_graph=False)
        #ew_model = copy.deepcopy(model)
        f_params_new = self.update_params_sgd(model, grads)


        for param_key, param in model.named_parameters():
            if param.requires_grad:
                param.data = f_params_new[param_key].data  # use data only as f_params_new has graph


        adv_logits = model(*vat_args)


        for param_key, param in model.named_parameters():
            if param.requires_grad:
                param.data = f_param[param_key]


        if task_type == TaskType.Ranking:
            adv_logits = adv_logits.view(-1, pairwise)
        #adv_lc = self.loss_map[task_id]

        #adv_loss = stable_kl(adv_logits, logits.detach())
        #mask = torch.max(F.softmax(logits, dim=-1), dim=-1)[0] > 0.8

        # adv_loss = adv_lc(logits[mask], adv_logits[mask], ignore_index=-1)
        #adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        adv_loss = torch.tensor([0]).to(logits.device)


        if torch.isnan(adv_loss):
            return torch.tensor([0]).to(logits.device), torch.tensor([0]), torch.tensor([0]), adv_logits
        #adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        #adv_loss = stable_kl(adv_logits, logits.detach())
        return adv_loss, torch.tensor([0]), torch.tensor([0]), adv_logits

    def update_params_sgd(self, model, grads):
        # supports SGD-like optimizers
        ans = {}
        for i, (name, para) in enumerate(model.named_parameters()):
            if para.requires_grad:
                #if self.adv:
                grad_direct = self._norm_grad(grads[i].detach())
                # else:
                #     grad_direct = para.data.new(para.size()).normal_(0, 1)
                #     grad_direct.detach()
                ans[name] = para - grad_direct * self.step_size * para.detach().abs().max()
        return ans
