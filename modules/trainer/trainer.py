from transformers import TrainingArguments as _hf_TrainingArguments
from dataclasses import dataclass, field
from typing import Optional
from fate.ml.nn.trainer.trainer_base import HomoTrainerMixin, FedArguments, get_ith_checkpoint
import os
import torch
import copy
from torch import nn
from typing import Any, Dict, List, Callable, Tuple, Union
from enum import Enum
from fate.arch import Context
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import Trainer as _hf_Trainer, EvalPrediction
from transformers.data.data_collator import DataCollator
from transformers.trainer_utils import has_length
from torch.utils.data import _utils
from transformers.trainer_callback import TrainerCallback
from typing import Optional
from dataclasses import dataclass, field
from transformers.modeling_utils import unwrap_model


TRAINABLE_WEIGHTS_NAME = "adapter_model.bin"


@dataclass
class _TrainingArguments(_hf_TrainingArguments):
    # in fate-2.0, we will control the output dir when using pipeline
    output_dir: str = field(default="./")
    disable_tqdm: bool = field(default=True)
    save_strategy: str = field(default="no")
    logging_strategy: str = field(default="epoch")
    logging_steps: int = field(default=1)
    evaluation_strategy: str = field(default="no")
    logging_dir: str = field(default=None)
    checkpoint_idx: int = field(default=None)
    # by default, we use constant learning rate, the same as FATE-1.X
    lr_scheduler_type: str = field(default="constant")
    log_level: str = field(default="info")
    deepspeed: Optional[str] = field(default=None)
    save_safetensors: bool = field(default=False)
    use_cpu: bool = field(default=False)

    def __post_init__(self):
        self.push_to_hub = False
        self.hub_model_id = None
        self.hub_strategy = "every_save"
        self.hub_token = None
        self.hub_private_repo = False
        self.push_to_hub_model_id = None
        self.push_to_hub_organization = None
        self.push_to_hub_token = None

        super().__post_init__()

DEFAULT_ARGS = _TrainingArguments().to_dict()

@dataclass
class TrainingArguments(_TrainingArguments):
    # To simplify the to dict result(to_dict only return non-default args)

    def to_dict(self):
        # Call the superclass's to_dict method
        all_args = super().to_dict()
        # Get a dict with default values for all fields
        default_args = copy.deepcopy(DEFAULT_ARGS)
        # Filter out args that are equal to their default values
        set_args = {name: value for name, value in all_args.items() if value != default_args.get(name)}
        return set_args

class Trainer(object):
    def __init__(
        self,
        model: torch.nn.Module,
        args: _hf_TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        save_trainable_weights_only: bool =  False
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model_init = model_init
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.optimizers = optimizers
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.save_trainable_weights_only = save_trainable_weights_only
    
    def train(self):
        trainer = _hf_Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            model_init=self.model_init,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
            optimizers=self.optimizers,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
        )
        
        trainer.train()
        self.model = unwrap_model(trainer.model)
        
        return trainer.state.log_history

    def save_model(
        self,
        output_dir: Optional[str] = None,
        state_dict=None
    ):
        if not self.save_trainable_weights_only:
            torch.save(self.model.state_dict(), output_dir + '/pytorch_model.bin')
        else:
            model = unwrap_model(self.model)

            if hasattr(model, "save_trainable"):
                model.save_trainable(output_dir)
            else:
                state_dict = {
                    k: p.to("cpu") for k,
                                       p in model.named_parameters() if p.requires_grad
                }

                torch.save(state_dict, output_dir + '/pytorch_model.bin')

class HomoTrainerClient(Trainer, HomoTrainerMixin):

    def __init__(
        self,
        ctx: Context,
        model: nn.Module,
        training_args: TrainingArguments,
        fed_args: FedArguments,
        train_set: Dataset,
        val_set: Dataset = None,
        optimizer: torch.optim.Optimizer = None,
        data_collator: Callable = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        local_mode: bool = False,
        save_trainable_weights_only: bool = False,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        # in case you forget to set evaluation_strategy
        if val_set is not None and training_args.evaluation_strategy == "no":
            training_args.evaluation_strategy = "epoch"

        HomoTrainerMixin.__init__(
            self,
            ctx=ctx,
            model=model,
            optimizer=optimizer,
            training_args=training_args,
            fed_args=fed_args,
            train_set=train_set,
            val_set=val_set,
            scheduler=scheduler,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
            local_mode=local_mode,
            save_trainable_weights_only=save_trainable_weights_only,
        )

        # concat checkpoint path if checkpoint idx is set
        if self._args.checkpoint_idx is not None:
            checkpoint_path = self._args.resume_from_checkpoint
            if checkpoint_path is not None and os.path.exists(checkpoint_path):
                checkpoint_folder = get_ith_checkpoint(checkpoint_path, self._args.checkpoint_idx)
                self._args.resume_from_checkpoint = os.path.join(checkpoint_path, checkpoint_folder)

        _hf_Trainer.__init__(
            self,
            model=model,
            args=self._args,
            train_dataset=train_set,
            eval_dataset=val_set,
            data_collator=data_collator,
            optimizers=(optimizer, scheduler),
            tokenizer=tokenizer,
            compute_metrics=self._compute_metrics_warp_func,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self._add_fate_callback(self.callback_handler)

    def _save(
        self,
        output_dir: Optional[str] = None,
        state_dict=None
    ):
        if not self._save_trainable_weights_only:
            return super()._save(output_dir, state_dict)
        else:
            model = unwrap_model(self.model)

            if hasattr(model, "save_trainable"):
                model.save_trainable(output_dir)
            else:
                state_dict = {
                    k: p.to("cpu") for k,
                                       p in model.named_parameters() if p.requires_grad
                }

                torch.save(state_dict, os.path.join(output_dir, TRAINABLE_WEIGHTS_NAME))