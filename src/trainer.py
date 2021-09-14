########## The following part is copied from Transformers' trainer (3.4.0) ##########

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the spe                                                                                                                                                                                                                                                                                               cific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
import gc
import collections
import inspect
import math
import os
import pdb
import re
import copy
import time
import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from models import LMForPromptFinetuning, BertForPromptFinetuning, RobertaForPromptFinetuning, DebertaForPromptFinetuning, resize_token_type_embeddings
import higher

import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from transformers.integrations import (
    default_hp_search_backend,
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    is_fairscale_available,
    deepspeed_init,
    is_deepspeed_zero3_enabled,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    get_tpu_sampler,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    speed_metrics,
    set_seed,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from dataset import patch_data
from tqdm import tqdm, trange

_use_native_amp = False
_use_apex = False
#
# DEFAULT_CALLBACKS = [DefaultFlowCallback]
# DEFAULT_PROGRESS_CALLBACK = ProgressCallback
#
# if is_in_notebook():
#     from transformers.utils.notebook import NotebookProgressCallback
#
#     DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback
#
# # Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast
#
# if version.parse(torch.__version__) < version.parse("1.2"):
#     _use_ddp_no_sync = False
# else:
#     _use_ddp_no_sync = True
#
# if is_datasets_available():
#     import datasets
#
# if is_torch_tpu_available():
#     import torch_xla.core.xla_model as xm
#     import torch_xla.debug.metrics as met
#     import torch_xla.distributed.parallel_loader as pl
#
# if is_tensorboard_available():
#     from transformers.integrations import TensorBoardCallback
#
#     DEFAULT_CALLBACKS.append(TensorBoardCallback)
#
#
# if is_wandb_available():
#     from transformers.integrations import WandbCallback
#
#     DEFAULT_CALLBACKS.append(WandbCallback)
#
# if is_comet_available():
#     from transformers.integrations import CometCallback
#
#     DEFAULT_CALLBACKS.append(CometCallback)
#
# if is_optuna_available():
#     import optuna
#
# if is_ray_available():
#     from ray import tune
#
# logger = logging.get_logger(__name__)

_is_torch_generator_available = False
_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
else:
    import torch.distributed as dist

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

if is_training_run_on_sagemaker():
    logging.add_handler(StreamHandler(sys.stdout))

if TYPE_CHECKING:
    import optuna
from accelerate import Accelerator, DistributedType

logger = logging.get_logger(__name__)


########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))


class Trainer(transformers.Trainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            train_demo_dataset: Optional[Dataset] = None,
            un_train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            **kwargs,
    ):
        if args is None:
            logger.info("No `TrainingArguments` passed, using the current path as `output_dir`.")
            args = TrainingArguments("tmp_trainer")
        self.args = args

        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)

        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False
        # Mixed precision setup
        self.use_apex = False
        self.use_amp = False
        self.fp16_backend = None
        self.current_flos = 0
        self._total_loss_scalar = 0.0
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        self.accelerator = Accelerator(fp16=self.args.fp16, cpu=self.args.cpu)

        assert (
                model is not None or model_init is not None
        ), "You must provide a model to use `Trainer`, either by using the `model` argument or the `model_init` argument."
        self.model_init = model_init
        if model is None and model_init is not None:
            model = self.call_model_init()
        #self.model = model.to(args.device) if model is not None else None
        #new add
        self.model = model.to(self.accelerator.device) if model is not None else None
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator

        #self.large_train_dataset = train_large_dataset
        #np.random.shuffle(train_dataset.example_idx)
        self.train_dataset = train_dataset
        self.train_demo_dataset = train_demo_dataset
        self.un_train_dataset = un_train_dataset
        # if self.args.use_last_epoch:
        #     self.eval_dataset = train_dataset
        # else:
        self.eval_dataset = eval_dataset

        self.tokenizer = tokenizer

        self.sharded_ddp = None
        if len(args.sharded_ddp) > 0:
            if args.deepspeed:
                raise ValueError(
                    "Using --sharded_ddp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )

            if args.local_rank == -1:
                raise ValueError("Using sharded DDP only works in distributed training.")
            elif not is_fairscale_available():
                raise ImportError("Sharded DDP training requires fairscale: `pip install fairscale`.")
            elif ShardedDDPOption.SIMPLE not in args.sharded_ddp and FullyShardedDDP is None:
                raise ImportError(
                    "Sharded DDP in a mode other than simple training requires fairscale version >= 0.3, found "
                    f"{fairscale.__version__}. Upgrade your fairscale library: `pip install --upgrade fairscale`."
                )
            elif ShardedDDPOption.SIMPLE in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.SIMPLE
            elif ShardedDDPOption.ZERO_DP_2 in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.ZERO_DP_2
            elif ShardedDDPOption.ZERO_DP_3 in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.ZERO_DP_3

        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers

        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument."
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks

        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Deprecated arguments
        if "tb_writer" in kwargs:
            warnings.warn(
                "Passing `tb_writer` as a keyword argument is deprecated and won't be possible in a "
                + "future version. Use `TensorBoardCallback(tb_writer=...)` instead and pass it to the `callbacks`"
                + "argument",
                FutureWarning,
            )
            tb_writer = kwargs.pop("tb_writer")
            self.remove_callback(TensorBoardCallback)
            self.add_callback(TensorBoardCallback(tb_writer=tb_writer))
        if "prediction_loss_only" in kwargs:
            warnings.warn(
                "Passing `prediction_loss_only` as a keyword argument is deprecated and won't be possible in a "
                + "future version. Use `args.prediction_loss_only` instead. Setting "
                + f"`args.prediction_loss_only={kwargs['prediction_loss_only']}",
                FutureWarning,
            )
            self.args.prediction_loss_only = kwargs.pop("prediction_loss_only")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create output directory if needed
        if self.is_world_process_zero():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_torch_tpu_available() and isinstance(self.model, PreTrainedModel):
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            self.data_collator = self.data_collator.collate_batch
            warnings.warn(
                (
                        "The `data_collator` should now be a simple callable (function, class with `__call__`), classes "
                        + "with a `collate_batch` are deprecated and won't be supported in a future version."
                ),
                FutureWarning,
            )

        if args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        # Enforce rules on using datasets with no __len__
        if train_dataset is not None and not isinstance(train_dataset, collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")
        if un_train_dataset is not None and not isinstance(un_train_dataset,
                                                           collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError("unlabeled train_dataset does not implement __len__, max_steps has to be specified")
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        if is_datasets_available():
            if isinstance(train_dataset, datasets.Dataset):
                self._remove_unused_columns(self.train_dataset, description="training")
            if isinstance(un_train_dataset, datasets.Dataset):
                self._remove_unused_columns(self.un_train_dataset, description="un_training")
            if isinstance(eval_dataset, datasets.Dataset):
                self._remove_unused_columns(self.eval_dataset, description="evaluation")

        self.state = TrainerState()
        self.control = TrainerControl()
        # Internal variable for total_flos used to count as tensors (for distributed + TPU), will be sent in the
        # state at each call to self.log.
        self._total_flos = None
        if self.args.fp16 and _use_native_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        default_label_names = (
            ["start_positions, end_positions"]
            if type(self.model) in MODEL_FOR_QUESTION_ANSWERING_MAPPING.values()
            else ["labels"]
        )
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        if args.fp16:
            if args.fp16_backend == "auto":
                self.fp16_backend = "amp" if _is_native_amp_available else "apex"
            else:
                self.fp16_backend = args.fp16_backend
            logger.info(f"Using {self.fp16_backend} fp16 backend")

        if args.fp16 and not args.deepspeed:  # deepspeed manages its own fp16
            if self.fp16_backend == "amp":
                self.use_amp = True
                if is_sagemaker_mp_enabled():
                    self.scaler = smp.amp.GradScaler()
                elif self.sharded_ddp is not None:
                    self.scaler = ShardedGradScaler()
                else:
                    self.scaler = torch.cuda.amp.GradScaler()
            else:
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to https://www.github.com/nvidia/apex."
                    )
                self.use_apex = True

    def create_optimizer_and_scheduler(self, num_training_steps, learning_rate=None, warmup_steps=None, weight_decay=None):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if learning_rate is None:
            learning_rate = self.args.learning_rate

        if warmup_steps is None:
            warmup_steps = self.args.warmup_steps

        if weight_decay is None:
            weight_decay = self.args.weight_decay

        if self.optimizer is None:
            params = {}
            prompt_params = {}
            for n, p in self.model.named_parameters():

                if 'prompt_embeddings' in n:
                    prompt_params[n] = p
                    continue
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    if p.requires_grad:
                        params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]


            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in prompt_params.items() if any(nd in n for nd in no_decay)],
                    "lr": self.args.prompt_learning_rate,
                }
            ]

            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        self.count_params()
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
            )


    def count_params(self):
        total_param = 0
        update_parameter = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            update_parameter.append(n)

            if 'prompt_embedding' in n:
                continue
            if 'decoder' in n:

                if self.model.data_args.task_name == 'mnli':
                    total_param += p.size(-1) * 3
                else:
                    total_param += p.size(-1) * 2
            else:
                total_param += p.numel()
            print(n, p.numel())
            #print(p.numel())



        print("Model parameters number is {}".format(total_param))
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behaviort.
        """

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs[0]

        return (loss, outputs) if return_outputs else loss

    def compute_contrast_loss(self, model, inputs, loss_weight=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if loss_weight is None:
            contra_loss = model.get_constrast_loss(**inputs)
        else:
            mask = loss_weight > 0
            inputs = {key: inputs[key][mask] for key in inputs}
            contra_loss = model.get_constrast_loss(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        return contra_loss

    def compute_adv_loss(self, model, inputs, loss_weight=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        NEED TO FIX
        """

        adv_loss = model.get_adv_loss(**inputs)
        if loss_weight is not None:
            adv_loss = (adv_loss.mean(-1) * loss_weight).sum()
        else:
            adv_loss = adv_loss.mean()

        return adv_loss

    def compute_un_loss(self, model, un_inputs, loss_weight=None, soft_label=None, fwd_type=None, features=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        NEED TO FIX
        # """
        # if self.label_smoother is not None and "psuedo_labels" in un_inputs:
        #     labels = un_inputs.pop("psuedo_labels")
        # else:
        #     labels = None

        un_labels = un_inputs['labels']
        del un_inputs['labels']

        outputs = model(**un_inputs)
        un_logits = outputs[0]

        if soft_label is not None:
            soft_label_way = soft_label == 1
        else:
            soft_label_way = (self.args.soft_label == 1)

        if soft_label_way:
            loss = F.kl_div(F.log_softmax(un_logits, dim=-1, dtype=torch.float32),
                            un_labels, reduction='none').sum(-1)
        else:
            loss = F.cross_entropy(un_logits, un_labels, reduction='none')

        if fwd_type is not None:
            if fwd_type == 4:
                sim = nn.CosineSimilarity(dim=-1)
                feature_loss = sim(features, outputs[1])
                loss = loss + feature_loss


        if loss_weight is not None:
            loss = (loss * loss_weight).sum()
        else:
            loss = loss.mean()


        return loss

    def meta_st(self, model, un_inputs, sampling_step=1, epsilon=1e-6, learning_rate=None, use_soft_label=True):

        model.eval()
        opti_param = [p for p in model.parameters() if p.requires_grad]
        if learning_rate is None:
            inner_opt = torch.optim.SGD(opti_param, lr=self.args.learning_rate)
        else:
            inner_opt = torch.optim.SGD(opti_param, lr=learning_rate)
        un_labels = un_inputs['labels']

        if use_soft_label is  None:
            use_soft_label = self.args.soft_label
            #un_labels = un_inputs['labels']

        # if self.args.fp16 and _use_native_amp:
        #     self.scaler.scale(loss).backward()
        # elif self.args.fp16 and _use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss.backward()

        for i in range(sampling_step):
            self.clear_memory()
            try:
                meta_inputs = next(self.meta_train_iter)
            except:
                self.meta_train_iter = self.meta_train_dataloader.__iter__()
                meta_inputs = next(self.meta_train_iter)
            #meta_inputs = self._prepare_inputs(meta_inputs)

            with higher.innerloop_ctx(
                    model, inner_opt, copy_initial_weights=True,
            ) as (fnet, diffopt):

                # fnet.lm_model = self.accelerator.prepare_model(fnet.lm_model)
                #diffopt = self.accelerator.prepare_optimizer(diffopt)

                # un_inputs['reduction'] = False

                un_outputs = fnet(**un_inputs)
                un_logits = un_outputs[1]

                if use_soft_label == 1:
                    un_loss = F.kl_div(F.log_softmax(un_logits, dim=-1, dtype=torch.float32),
                                    un_labels, reduction='none').sum(-1)
                else:
                    un_loss = F.cross_entropy(un_logits, un_labels, reduction='none')
                weight = torch.zeros(un_loss.size(), requires_grad=True).to(self.accelerator.device)
                new_loss = (un_loss * weight).sum()

                diffopt.step(new_loss)

                meta_outputs = fnet(**meta_inputs)
                loss = meta_outputs["loss"] if isinstance(meta_outputs, dict) else meta_outputs[0]

                grad_eps = torch.autograd.grad(loss, weight, only_inputs=True)[0].detach()

            if i == 0:
                loss_weight = (- grad_eps)
            else:
                loss_weight += (-grad_eps)

            model.zero_grad()

        loss_weight = torch.clamp(loss_weight, min=0)

        #loss_weight = (loss_weight > 0) + 0
        norm_c = torch.sum(loss_weight) + epsilon
        if norm_c != 0:
            loss_weight = loss_weight / norm_c
        else:
            loss_weight = loss_weight
        model.train()
        return loss_weight.detach()

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def pseudo_data_selection(self, model, un_inputs, un_meta, learning_rate=None, use_soft_label=None):
        loss_weight = None

        if self.args.psuedo_selection_opt == "meta_st" and self.delta > self.meta_st_warmup_steps:
            loss_weight = self.meta_st(model, un_inputs, sampling_step=self.args.sampling_steps, learning_rate=learning_rate, use_soft_label=use_soft_label)

        elif self.args.psuedo_selection_opt == "confidence" or (self.delta < self.meta_st_warmup_steps and  self.args.psuedo_selection_opt == "meta_st"):
            soft_labels = un_meta.get("soft_labels", None)
            mask = torch.max(soft_labels, dim=-1)[0] > self.args.confidence_thresh
            loss_weight = mask / (mask.sum() + 1e-5)

        return loss_weight

    def assign_psuedo_label(self, un_inputs, model=None, soft_label=None, fwd_type=None):


        if model is None:
            teacher_model = self.teacher_model.to(self.accelerator.device)
            teacher_model.eval()
        else:
            teacher_model = model
            teacher_model.eval()


        with torch.no_grad():
            outputs = teacher_model(**un_inputs)

            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            # NEED TO FIX

            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            logits = logits.detach()

            if fwd_type is not None and fwd_type == 4:
                features = outputs[1].detach()

            soft_labels = F.softmax(logits, dim=-1)
            hard_labels = torch.max(F.softmax(logits, dim=-1), dim=-1)[1]

            if soft_label is not None:
                use_soft_label = (soft_label == 1)
            else:
                use_soft_label = (self.args.soft_label == 1)
            if use_soft_label == 1:
                if self.args.sharpen == 1:
                    sharpen_soft_labels = logits / self.args.temperature
                    un_inputs['labels'] = F.softmax(sharpen_soft_labels, dim=-1)
                else:
                    un_inputs['labels'] = soft_labels

            else:
                un_inputs['labels'] = hard_labels

        self.clear_memory()
        if fwd_type is not None and fwd_type == 4:
            return soft_labels, hard_labels, features

        return soft_labels, hard_labels

    def get_un_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler
        (adapted to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.un_train_dataset is None:
            raise ValueError("Trainer: un_training requires a un_train_dataset.")
        un_train_sampler = self._get_un_train_sampler()

        return DataLoader(
            self.un_train_dataset,
            batch_size=self.args.un_train_batch_size,
            sampler=un_train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_demo_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler
        (adapted to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_demo_dataset is None:
            raise ValueError("Trainer: un_training requires a un_train_dataset.")
        train_demo_sampler = self._get_train_demo_sampler()

        return DataLoader(
            self.train_demo_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_demo_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_meta_train_dataloader(self, percentage=1) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler
        (adapted to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: un_training requires a un_train_dataset.")
        train_sampler = self._get_train_sampler()

        if percentage == 1:
            self.meta_train_dataset = self.train_dataset
        else:
            self.meta_dataset = copy.deepcopy(self.train_dataset)
            self.meta_train_dataset = copy.deepcopy(self.train_dataset)
            np.random.shuffle(self.meta_dataset.example_idx)
            self.meta_train_num = int(percentage * len(self.meta_dataset.example_idx))
            self.meta_train_dataset.example_idx = self.meta_dataset.example_idx[:self.meta_train_num]



        return DataLoader(
            self.meta_train_dataset,
            batch_size=self.args.meta_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_meta_valid_dataloader(self, percentage=1) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler
        (adapted to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: un_training requires a un_train_dataset.")
        train_sampler = self._get_train_sampler()

        if percentage == 1:
            self.meta_valid_dataset = self.train_dataset
        else:
            self.meta_valid_dataset = copy.deepcopy(self.train_dataset)
            self.meta_valid_dataset.example_idx = self.meta_dataset.example_idx[self.meta_train_num:]


        return DataLoader(
            self.meta_valid_dataset,
            batch_size=self.args.meta_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )


    def _get_un_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.un_train_dataset, collections.abc.Sized):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.un_train_dataset)
        else:
            return (
                RandomSampler(self.un_train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.un_train_dataset)
            )

    def _get_train_demo_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.train_demo_dataset, collections.abc.Sized):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_demo_dataset)
        else:
            return (
                RandomSampler(self.train_demo_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_demo_dataset)
            )

    def demon_condensation_step(self, model, demo_inputs=None, inputs=None):


        if inputs is not None:
            del inputs['labels']
        loss = 0
        loss_weight=None
        un_meta = {}
        fwd_type = -1
        demo_inputs['fwd_type'] = fwd_type
        inputs['fwd_type'] = fwd_type
        features = None

        if fwd_type == 4:
            soft_labels, hard_labels, features = self.assign_psuedo_label(demo_inputs, model=model, soft_label=1, fwd_type=fwd_type)
            inputs['labels'] = soft_labels
        else:
            soft_labels, hard_labels = self.assign_psuedo_label(demo_inputs, model=model, soft_label=1)
            inputs['labels'] = soft_labels


        # un_meta['soft_labels'] = soft_labels
        # un_meta['hard_labels'] = hard_labels
        # if self.args.soft_label == 1:
        #     un_meta['pseudo_label'] = soft_labels
        # else:
        #     un_meta['pseudo_label'] = hard_labels

        model.train()
        un_meta['soft_labels'] = soft_labels
        un_meta['hard_labels'] = hard_labels


        del inputs['fwd_type']

        loss_weight = self.pseudo_data_selection(model, inputs, un_meta, use_soft_label=1)

        inputs['fwd_type'] = fwd_type
        conden_loss = self.compute_un_loss(model, inputs, loss_weight=loss_weight, soft_label=1, fwd_type=fwd_type, features=features)
        loss = loss + conden_loss

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # if self.args.fp16 and _use_native_amp:
        #     self.scaler.scale(loss).backward()
        # elif self.args.fp16 and _use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #    loss.backward()
        if isinstance(loss, int):
            return 0
        self.accelerator.backward(loss)
        return loss.detach().item()



    def training_step(self, model, inputs=None, un_inputs=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        un_meta = {}
        model.train()

        # if inputs is not None:
        #     inputs = self._prepare_inputs(inputs)

        if un_inputs is not None:

            del un_inputs['labels']
            #un_inputs = self._prepare_inputs(un_inputs)
        loss = 0

        if inputs is not None:
            if self.loss_alpha != 0:
                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        loss = self.compute_loss(model, inputs)
                else:
                    loss = self.compute_loss(model, inputs)

            if self.args.adv_opt == 1 or self.args.adv_opt == 3:
                adv_loss = self.compute_adv_loss(model, inputs)
                loss = loss + adv_loss

            if self.args.contrast_training == 1:
                contrast_loss = self.compute_contrast_loss(model, inputs)
                # print(contrast_loss)
                loss = loss + contrast_loss

        if self.args.is_semi == 1 and (un_inputs is not None) and (len(un_inputs) != 0):
            if self.args.use_psuedo_label > 0:

                soft_labels, hard_labels = self.assign_psuedo_label(un_inputs)
                un_meta['soft_labels'] = soft_labels
                un_meta['hard_labels'] = hard_labels
                if self.args.soft_label == 1:
                    un_meta['pseudo_label'] = soft_labels
                else:
                    un_meta['pseudo_label'] = hard_labels

                loss_weight = self.pseudo_data_selection(model, un_inputs, un_meta)
                un_loss = self.compute_un_loss(model, un_inputs, loss_weight=loss_weight)
                loss = loss + un_loss

            if self.args.adv_opt > 1:
                un_adv_loss = self.compute_adv_loss(model, un_inputs, loss_weight=loss_weight)
                loss = loss + un_adv_loss

            if self.args.contrast_training > 1:
                un_inputs['labels'] = un_meta['hard_labels']
                contrast_loss = self.compute_contrast_loss(model, un_inputs, loss_weight=loss_weight)
                # print(contrast_loss)
                loss = loss + contrast_loss

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # if self.args.fp16 and _use_native_amp:
        #     self.scaler.scale(loss).backward()
        # elif self.args.fp16 and _use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #    loss.backward()
        if isinstance(loss, int):
            return 0
        self.accelerator.backward(loss)

        return loss.detach().item()

    def update_teacher(self, model):

        model_file = os.path.join(self.args.output_dir, "pytorch_model.bin")
        if self.teacher_model is None:
            self.teacher_model = copy.deepcopy(model)

        if self.args.mean_teacher:
            pass
        else:
            if model_file is not None and os.path.exists(model_file):
                logger.info('loading model from {}'.format(model_file))
                # if self.args.n_gpu > 1:
                #     self.teacher_model = torch.nn.DataParallel(self.teacher_model)

                self.teacher_model.load_state_dict(torch.load(model_file))


    def re_init(self):
        config = AutoConfig.from_pretrained(
            self.teacher_model.model_args.config_name if self.teacher_model.model_args.config_name else self.teacher_model.model_args.model_name_or_path,
            num_labels=self.teacher_model.num_labels,
            finetuning_task=self.teacher_model.data_args.task_name,
            cache_dir=self.teacher_model.model_args.cache_dir,
        )

        if 'prompt' in self.teacher_model.model_args.few_shot_type:
            # if config.model_type == 'roberta':
            #     model_fn = RobertaForPromptFinetuning
            # elif config.model_type == 'bert':
            #     model_fn = BertForPromptFinetuning
            # elif config.model_type =='deberta':
            #     model_fn = DebertaForPromptFinetuning
            model_fn = LMForPromptFinetuning
            # else:
            #     raise NotImplementedError
        elif self.teacher_model.model_args.few_shot_type == 'finetune':
            model_fn = AutoModelForSequenceClassification
        else:
            raise NotImplementedError


        if self.teacher_model.data_args.prompt:
            self.model = model_fn(config, self.teacher_model.model_args, self.teacher_model.data_args)
        else:
            self.model = model_fn.from_pretrained(
                self.teacher_model.model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self.teacher_model.model_args.model_name_or_path),
                config=config,
                cache_dir=self.teacher_model.model_args.cache_dir,
            )
            self.model.model_args = self.teacher_model.model_args
            self.model.data_args = self.teacher_model.data_args
        self.wipe_memory()


        # self.model = self.model.from_pretrained(
        #     self.teacher_model.model_args.model_name_or_path,
        #     from_tf=bool(".ckpt" in self.teacher_model.model_args.model_name_or_path),
        #     config=config,
        #     cache_dir=self.teacher_model.model_args.cache_dir,
        # )


        ##new add


        #self.model = self.model.to(self.args.device) if self.model is not None else None

        # if self.args.fp16 and _use_apex:
        #     if not transformers.is_apex_available():
        #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        #     self.model, optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.args.fp16_opt_level)
        #
        #     # Multi-gpu training (should be after apex fp16 initialization)
        # if self.args.n_gpu > 1:
        #     self.model = torch.nn.DataParallel(self.model)
        #
        #     # Distributed training (should be after apex fp16 initialization)
        # if self.args.local_rank != -1:
        #     self.model = torch.nn.parallel.DistributedDataParallel(
        #         self.model,
        #         device_ids=[self.args.local_rank],
        #         output_device=self.args.local_rank,
        #         find_unused_parameters=True,
        #     )

        # if self.teacher_model.data_args.prompt:
        #     self.model.label_word_list = self.teacher_model.label_word_list
        # if output_modes_mapping[data_args.task_name] == 'regression':
        #     # lower / upper bounds
        #     model.lb, model.ub = bound_mapping[data_args.task_name]
        # self.model.model_args = self.teacher_model.model_args
        # self.model.data_args = self.teacher_model.data_args
        # self.model.tokenizer = self.teacher_model.tokenizer

    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def wipe_memory(self):  # DOES WORK
        self._optimizer_to(torch.device('cpu'))
        del self.optimizer
        self.clear_memory()
        self.optimizer=None

    def _optimizer_to(self, device):
        for param in self.optimizer.state_dict()['state'].values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)



  


    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """


        self._memory_tracker.start()

        self.teacher_model = None
        self.best_dir = None
        args = self.args
        self.is_in_train = True
        self.objective = -float("inf")
        start_time = time.time()
        self.state.max_steps = self.args.max_steps
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()

        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps

        self_training_start_iter = self.args.self_training_start_epoch * int(
            len(train_dataloader) // self.args.gradient_accumulation_steps)

        finetune_teacher_steps = self.args.finetune_teacher_epoch * int(
            len(train_dataloader) // self.args.gradient_accumulation_steps)

        update_teacher_steps = (self.args.update_teacher_steps // self.args.un_gradient_accumulation_steps)

        if self.args.is_semi == 1:
            self_training_total_steps = update_teacher_steps + finetune_teacher_steps
            self.un_train_dataloader = self.get_un_train_dataloader()
        else:
            self.un_train_dataloader = None
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        elif self.args.is_semi == 1 and self.args.self_training_session > 0:
            t_total =  self_training_total_steps * self.args.self_training_session + self_training_start_iter
            self.args.max_steps = t_total
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs


        self.meta_st_warmup_steps = int(self.args.meta_st_warmup * self.args.update_teacher_steps)

        ######TEMP CHANGE
        if self.args.freeze_encoder: # and not (self.args.psuedo_selection_opt == 'meta_st' and self.args.is_semi == 1):
            logger.info("Freeze language model encoder")
            self.model.freeze_lm_encoder()
        ######TEMP CHANGE
        if self.args.only_train_bias: # and not (self.args.psuedo_selection_opt == 'meta_st' and self.args.is_semi == 1):
            logger.info("only finetune bias")
            self.model.freeze_lm_finetune_bias()

        if self.args.update_k_layers != -1:
            self.model.freeeze_lm_k_layers(self.args.update_k_layers)

        if self.args.update_component != "none":
            self.model.freeze_lm_component(self.args.update_component)


        if self.args.is_semi == 1:
            self.create_optimizer_and_scheduler(num_training_steps=self_training_start_iter)
        else:
            self.create_optimizer_and_scheduler(num_training_steps=t_total)
        self.t_total = t_total
        self.meta_train_dataloader = self.get_meta_train_dataloader()
        self.meta_valid_dataloader = self.get_meta_valid_dataloader()


        un_inputs = None


        ## new add
        self.train_dataloader = train_dataloader
        train_dataloader = self.get_train_dataloader()
        self.demo_train_dataloader = self.get_demo_train_dataloader()

        self.model, self.optimizer, self.train_dataloader,  self.meta_train_dataloader, self.meta_valid_dataloader, self.un_train_dataloader, self.demo_train_dataloader\
            = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader,  self.meta_train_dataloader, self.meta_valid_dataloader, self.un_train_dataloader, self.demo_train_dataloader
        )

        un_train_dataloader = self.un_train_dataloader
        train_dataloader = self.train_dataloader
        demo_train_dataloader = self.demo_train_dataloader

        self.meta_train_iter = self.meta_train_dataloader.__iter__()


        # if self.args.start_from_freeze and args.freeze:
        #     self.model.freeze_lm()





        # optimizer = self.optimizer
        # scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
                model_path is not None
                and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))




        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                    self.args.train_batch_size
                    * self.args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Unlabeled Gradient Accumulation steps = %d", self.args.un_gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        self.loss_alpha = 1
        epochs_trained = 0
        finetune = True
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                        len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        self.model.zero_grad()
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch"
        )
        demo_train_iter = None
        demo_inputs=None
        session_num = 0



        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader


            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            # steps_in_epoch = (
            #     len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            # )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            un_train_iter = None




            for step, inputs in enumerate(epoch_iterator):
                delta = (self.global_step - self_training_start_iter)

                if self.args.is_semi == 1 and (delta >= 0):
                    # if delta == 0:
                    #     self.objective = -float("inf")
                    delta = delta % self_training_total_steps
                    self.delta = delta

                    if self.args.psuedo_selection_opt == 'meta_st' or self.args.semi_finetune:
                        if delta >= update_teacher_steps:
                            if delta == update_teacher_steps:

                                logger.info("######### Start finetuning #########")
                                # self.args.learning_rate = learning_rate
                                output = self.evaluate()
                                metrics = output.metrics
                                objective = self.dev_objective(metrics)
                                logger.info("Test result: {}".format(objective))
                                # self.optimizer = None
                                # self.lr_scheduler = None
                                # t_total = finetune_teacher_steps
                                #
                                #
                                # self.create_optimizer_and_scheduler(num_training_steps=t_total)
                                # optimizer = self.optimizer
                                # scheduler = self.lr_scheduler
                            finetune = True

                            self.loss_alpha = 1
                        else:
                            if delta == 0 and (self.args.semi_finetune or self.args.psuedo_selection_opt == 'meta_st'):
                                if self.args.psuedo_selection_opt == 'meta_st':
                                    logger.info("######### Start meta st #########")
                                if self.args.use_last_epoch:
                                    output = self.evaluate()
                                    metrics = output.metrics
                                    objective = self.dev_objective(metrics)
                                    self.objective = -float("inf")
                                    logger.info("Test result: {}".format(objective))
                                    self.save_model(self.args.output_dir)
                                session_num += 1
                                if session_num > self.args.self_training_session:
                                    break
                            finetune = False
                            self.loss_alpha = 0

                    if delta == 0:
                        self.model.zero_grad()
                        if self.args.use_psuedo_label > 0:
                            self.update_teacher(self.model)

                        if self.args.re_init or self.args.psuedo_selection_opt == 'meta_st':

                            logger.info('##### RE INIT MODEL #########')
                            self.re_init()


                            if self.args.freeze_encoder:
                                logger.info("Freeze language model encoder")
                                self.model.freeze_lm_encoder()

                            elif self.args.only_train_bias:
                                logger.info("only finetune bias")
                                self.model.freeze_lm_finetune_bias()

                            elif self.args.update_component != "none":
                                self.model.freeze_lm_component(self.args.update_component)

                            semi_learning_rate = self.args.semi_learning_rate
                            semi_warmup_steps = self.args.semi_warmup_ratio * update_teacher_steps
                            semi_weight_decay = self.args.semi_weight_decay
                            # self.args.learning_rate = self.args.prompt_learning_rate
                            self.optimizer = None
                            self.lr_scheduler = None
                            t_total = self_training_total_steps
                            self.create_optimizer_and_scheduler(num_training_steps=t_total, learning_rate=semi_learning_rate, warmup_steps=semi_warmup_steps, weight_decay=semi_weight_decay)
                            self.t_total = t_total
                            #####
                            #self.train_dataset = self.normal_train_dataset
                            self.train_dataloader = self.accelerator.prepare(self.get_train_dataloader())
                            epoch_iterator = self.train_dataloader
                            #####
                            # self.model.lm_model, self.optimizer = self.accelerator.prepare(
                            #     self.model.lm_model, self.optimizer
                            # )

                            self.model = self.model.to(self.accelerator.device)




                    if un_train_iter is None:
                        un_train_iter = un_train_dataloader.__iter__()

                    try:
                        un_inputs = next(un_train_iter)
                    except:
                        un_train_iter = un_train_dataloader.__iter__()
                        un_inputs = next(un_train_iter)


                if self.args.demo_condon == 1:
                    demo_inputs = patch_data(self.train_demo_dataset, inputs)


                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if un_inputs is not None and (not finetune):
                    try:
                        tr_loss += self.training_step(self.model,  un_inputs=un_inputs)
                    except:
                        print("One error here")
                        self.clear_memory()
                    tr_loss += self.training_step(self.model, inputs=inputs)
                elif demo_inputs is not None and self.args.demo_condon == 1:
                    tr_loss += self.demon_condensation_step(self.model, demo_inputs=demo_inputs, inputs=inputs)
                else:
                    tr_loss += self.training_step(self.model, inputs)
                self.current_flos += float(self.floating_point_ops(inputs))

                if (finetune and (step + 1) % self.args.gradient_accumulation_steps == 0) or (not finetune and (step + 1) % self.args.un_gradient_accumulation_steps == 0) or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):

                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["norm"] = norm.item()
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)

                    # ----------------------------------------------------------------------
                    # BEGIN CHANGES.
                    # ----------------------------------------------------------------------

                    if not finetune and self.global_step % self.args.eval_steps == 0:  ####temp change
                        output = self.evaluate()
                        metrics = output.metrics
                        objective = self.dev_objective(metrics)
                        logger.info("Test result: {}".format(objective))

                    metrics = None
                    if self.args.use_last_epoch: # and not (self.args.semi_finetune and self.args.is_semi == 1):
                        continue
                    if self.global_step % self.args.eval_steps == 0:
                        if self.args.psuedo_selection_opt == 'meta_st' and \
                            self.args.is_semi == 1 and not (finetune):
                            output = self.evaluate(self.meta_valid_dataset)
                        elif self.args.semi_finetune and \
                                self.args.is_semi == 1 and not (finetune):
                            output = self.evaluate(self.train_dataset)
                        elif self.args.use_last_epoch:
                            continue
                        else:
                            output = self.evaluate()

                        metrics = output.metrics

                        objective = self.dev_objective(metrics)
                        logger.info("Dev result: {}".format(objective))


                        if objective > self.objective:
                            logger.info("Best dev result: {}".format(objective))
                            self.objective = objective
                            self.save_model(self.args.output_dir)

                            # ----------------------------------------------------------------------
                    # END CHANGES.
                    # ----------------------------------------------------------------------


                if self.args.max_steps > 0 and self.global_step > self.args.max_steps or \
                    (session_num > self.args.self_training_session and self.args.is_semi == 1):
                    break

            if self.args.use_last_epoch:  # and not (self.args.semi_finetune and self.args.is_semi == 1):
                continue

            if self.args.is_semi == 1:
                continue

            output = self.evaluate()
            metrics = output.metrics

            objective = self.dev_objective(metrics)
            logger.info("Dev result: {}".format(objective))

            if objective > self.objective:
                logger.info("Best dev result: {}".format(objective))
                self.objective = objective
                self.save_model(self.args.output_dir)

            if self.args.max_steps > 0 and self.global_step > self.args.max_steps or \
                    (session_num > self.args.self_training_session and self.args.is_semi == 1):
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        if self.args.use_last_epoch:
            self.save_model(self.args.output_dir)

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        self.log(metrics)
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.state.global_step, metrics)
        # return TrainOutput(self.global_step, tr_loss / self.global_step), self.objective

    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
