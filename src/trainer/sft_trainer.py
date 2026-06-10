import os
import torch
import torch.nn as nn
from typing import Optional, List, Union, Dict, Any
from dataclasses import dataclass

from transformers import Trainer, GenerationConfig
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy,
    has_length,
)
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS
)
from transformers.trainer_utils import EvalLoopOutput
from torch.utils.data import DataLoader
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

from constants import IGNORE_INDEX


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


@dataclass
class GenerativeEvalPrediction:
    """Container for generative evaluation predictions."""
    predictions: List[str]
    references: List[str]


class QwenSFTTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(QwenSFTTrainer, self).__init__(*args, **kwargs)
        # processing_class is set by parent Trainer from the constructor argument
        # We can access it via self.processing_class (same as processor)

    def compute_cfg_loss(self, model, inputs, is_dropped, return_outputs=False, **kwargs):
        """Compute the Classifier-Free Guidance (CFG) dual loss."""
        cfg_weight = getattr(self.args, "cfg_loss_weight", 0.0)
        cfg_drop_prob = getattr(self.args, "cfg_drop_prob", 0.0)
        
        # Fast path: CFG completely disabled — use model's internal CE
        if cfg_weight == 0.0 or cfg_drop_prob == 0.0 or is_dropped is None:
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        num_dropped = is_dropped.sum().item()
        batch_size = is_dropped.size(0)

        # Homogeneous batches (purely conditional or purely unconditional).
        # This is ALWAYS true when batch_size == 1 per device.
        # It keeps Liger Kernel fully active since we pass labels, avoiding materializing logits.
        if num_dropped == 0 or num_dropped == batch_size:
            outputs = model(**inputs)
            loss_unscaled = outputs.loss.item()
            loss = outputs.loss

            if num_dropped == 0:
                # Purely conditional batch: scale for expectation correction
                loss = loss / (1.0 - cfg_drop_prob)
                loss_cond_val = f"{loss_unscaled:.6f}"
                loss_uncond_val = "N/A"
            else:
                # Purely unconditional batch: scale for expectation correction
                loss = -cfg_weight * loss / cfg_drop_prob
                loss_cond_val = "N/A"
                loss_uncond_val = f"{loss_unscaled:.6f}"

            print(f"\n[GPU {self.accelerator.process_index}] [CFG Loss] Samples Dropped: {num_dropped} / {batch_size}")
            print(f"  Conditional Loss:   {loss_cond_val}")
            print(f"  Unconditional Loss: {loss_uncond_val}")
            print(f"  Final Loss:         {loss.item():.6f}\n")

            return (loss, outputs) if return_outputs else loss

        # Fallback for mixed batches (only possible when batch_size > 1).
        # We manually compute shifted CE to calculate cond/uncond separately.
        # We process sample-by-sample to avoid contiguous() allocations of the full batch logits.
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits             # (B, seq_len, vocab_size)

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        per_sample_loss = []

        for i in range(batch_size):
            # Slicing the first dimension of a contiguous 3D tensor returns a contiguous 2D tensor.
            shift_logits_i = logits[i, :-1, :]  # shape: (seq_len - 1, vocab_size)
            shift_labels_i = labels[i, 1:]     # shape: (seq_len - 1)

            loss_i = loss_fct(shift_logits_i, shift_labels_i)
            mask_i = shift_labels_i != IGNORE_INDEX
            mean_loss_i = loss_i[mask_i].sum() / mask_i.sum().clamp(min=1)
            per_sample_loss.append(mean_loss_i)

        per_sample_loss = torch.stack(per_sample_loss)

        cond_mask = ~is_dropped
        uncond_mask = is_dropped

        loss_cond = per_sample_loss[cond_mask].mean() if cond_mask.any() else torch.tensor(0.0, device=logits.device)
        loss_uncond = per_sample_loss[uncond_mask].mean() if uncond_mask.any() else torch.tensor(0.0, device=logits.device)

        # Scale each part by its probability for correct expected value
        loss = (loss_cond / (1.0 - cfg_drop_prob)) + cfg_weight * (-loss_uncond / cfg_drop_prob)

        loss_cond_val = f"{loss_cond.item():.6f}" if cond_mask.any() else "N/A"
        loss_uncond_val = f"{loss_uncond.item():.6f}" if uncond_mask.any() else "N/A"
        print(f"\n[GPU {self.accelerator.process_index}] [CFG Loss] Samples Dropped: {num_dropped} / {batch_size}")
        print(f"  Conditional Loss:   {loss_cond_val}")
        print(f"  Unconditional Loss: {loss_uncond_val}")
        print(f"  Final Loss:         {loss.item():.6f}\n")

        return (loss, outputs) if return_outputs else loss


    def compute_cfg_margin_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute the CFG margin loss: max(0, margin - log(P(y|img, txt)/P(y|dropped, txt)))."""
        margin = getattr(self.args, "cfg_loss_margin", 1.0)
        batch_size = inputs["input_ids"].size(0)

        # Clone inputs to create the unconditional inputs (zeroed out images)
        import copy
        inputs_uncond = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs_uncond[k] = v.clone()
            else:
                inputs_uncond[k] = copy.deepcopy(v)

        # Zero out pixel values for the unconditional pass
        if "pixel_values" in inputs_uncond:
            inputs_uncond["pixel_values"] = torch.zeros_like(inputs_uncond["pixel_values"])

        # Unconditional pass: Run with no_grad to avoid keeping activations in memory
        with torch.no_grad():
            if batch_size == 1:
                outputs_uncond = model(**inputs_uncond)
                loss_uncond = outputs_uncond.loss
            else:
                # Fallback for batch_size > 1
                labels = inputs.get("labels")
                outputs_uncond = model(**inputs_uncond)
                logits_uncond = outputs_uncond.logits

                loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
                per_sample_loss_uncond = []

                for i in range(batch_size):
                    shift_logits_uncond_i = logits_uncond[i, :-1, :]
                    shift_labels_i = labels[i, 1:]
                    mask_i = shift_labels_i != IGNORE_INDEX
                    num_tokens_i = mask_i.sum().clamp(min=1)

                    loss_uncond_i = loss_fct(shift_logits_uncond_i, shift_labels_i)
                    mean_loss_uncond_i = loss_uncond_i[mask_i].sum() / num_tokens_i
                    per_sample_loss_uncond.append(mean_loss_uncond_i)

                per_sample_loss_uncond = torch.stack(per_sample_loss_uncond)
                loss_uncond = per_sample_loss_uncond

        # Conditional pass: Run with gradients enabled (activations cached for backward)
        if batch_size == 1:
            outputs_cond = model(**inputs)
            loss_cond = outputs_cond.loss

            # Detach loss_uncond to avoid keeping its graph
            loss = torch.clamp(margin - loss_uncond.detach() + loss_cond, min=0.0)
            print(f"\n[GPU {self.accelerator.process_index}] [CFG Margin Loss]")
            print(f"  Conditional Loss:   {loss_cond.item():.6f}")
            print(f"  Unconditional Loss: {loss_uncond.item():.6f}")
            print(f"  Final Loss:         {loss.item():.6f}\n")
            return (loss, outputs_cond) if return_outputs else loss

        # Fallback for batch_size > 1
        labels = inputs.pop("labels")
        outputs_cond = model(**inputs)
        logits_cond = outputs_cond.logits

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        per_sample_loss_cond = []

        for i in range(batch_size):
            shift_logits_cond_i = logits_cond[i, :-1, :]
            shift_labels_i = labels[i, 1:]
            mask_i = shift_labels_i != IGNORE_INDEX
            num_tokens_i = mask_i.sum().clamp(min=1)

            loss_cond_i = loss_fct(shift_logits_cond_i, shift_labels_i)
            mean_loss_cond_i = loss_cond_i[mask_i].sum() / num_tokens_i
            per_sample_loss_cond.append(mean_loss_cond_i)

        per_sample_loss_cond = torch.stack(per_sample_loss_cond)

        # Detach loss_uncond to avoid keeping its graph
        loss = torch.clamp(margin - per_sample_loss_uncond.detach() + per_sample_loss_cond, min=0.0).mean()
        print(f"\n[GPU {self.accelerator.process_index}] [CFG Margin Loss]")
        print(f"  Conditional Loss:   {per_sample_loss_cond.mean().item():.6f}")
        print(f"  Unconditional Loss: {per_sample_loss_uncond.mean().item():.6f}")
        print(f"  Final Loss:         {loss.item():.6f}\n")
        return (loss, outputs_cond) if return_outputs else loss
    def compute_cfg_conf_reg_loss(self, model, inputs, is_dropped, return_outputs=False, **kwargs):
        """Compute the CFG confidence regularization loss."""
        cfg_weight = getattr(self.args, "cfg_loss_weight", 0.0)
        cfg_drop_prob = getattr(self.args, "cfg_drop_prob", 0.0)
        reg_weight = getattr(self.args, "cfg_reg_weight", 0.0)

        # Fast path: CFG completely disabled — use model's internal CE
        if cfg_weight == 0.0 or is_dropped is None or cfg_drop_prob == 0.0:
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        batch_size = is_dropped.size(0)

        # Pop labels to prevent the model from computing cross entropy internally.
        # This guarantees outputs.logits is returned (not None) and ensures all ranks
        # run the same forward graph to prevent distributed deadlocks/hanging.
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        per_sample_loss = []
        per_sample_entropy = []

        for i in range(batch_size):
            shift_logits_i = logits[i, :-1, :]
            shift_labels_i = labels[i, 1:]
            mask_i = shift_labels_i != IGNORE_INDEX
            num_tokens_i = mask_i.sum().clamp(min=1)

            # Cross entropy loss
            loss_i = loss_fct(shift_logits_i, shift_labels_i)
            mean_loss_i = loss_i[mask_i].sum() / num_tokens_i
            per_sample_loss.append(mean_loss_i)

            # Entropy H
            log_probs_i = torch.log_softmax(shift_logits_i, dim=-1)
            probs_i = torch.softmax(shift_logits_i, dim=-1)
            entropy_i = -(probs_i * log_probs_i).sum(dim=-1)
            mean_entropy_i = entropy_i[mask_i].sum() / num_tokens_i
            per_sample_entropy.append(mean_entropy_i)

        per_sample_loss = torch.stack(per_sample_loss)
        per_sample_entropy = torch.stack(per_sample_entropy)

        cond_mask = ~is_dropped
        uncond_mask = is_dropped

        loss_cond = per_sample_loss[cond_mask].mean() if cond_mask.any() else torch.tensor(0.0, device=logits.device)
        loss_uncond = per_sample_loss[uncond_mask].mean() if uncond_mask.any() else torch.tensor(0.0, device=logits.device)
        entropy_uncond = per_sample_entropy[uncond_mask].mean() if uncond_mask.any() else torch.tensor(0.0, device=logits.device)

        # Scale each part by its probability for correct expected value
        loss = (loss_cond / (1.0 - cfg_drop_prob)) + cfg_weight * (-loss_uncond / cfg_drop_prob) - reg_weight * (entropy_uncond / cfg_drop_prob)

        num_dropped = is_dropped.sum().item()
        loss_cond_val = f"{loss_cond.item():.6f}" if cond_mask.any() else "N/A"
        loss_uncond_val = f"{loss_uncond.item():.6f}" if uncond_mask.any() else "N/A"
        entropy_uncond_val = f"{entropy_uncond.item():.6f}" if uncond_mask.any() else "N/A"
        print(f"\n[GPU {self.accelerator.process_index}] [CFG Conf Reg Loss] Samples Dropped: {num_dropped} / {batch_size}")
        print(f"  Conditional Loss:   {loss_cond_val}")
        print(f"  Unconditional Loss: {loss_uncond_val}")
        print(f"  Uncond Entropy:     {entropy_uncond_val}")
        print(f"  Final Loss:         {loss.item():.6f}\n")

        return (loss, outputs) if return_outputs else loss


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override to support CFG dual loss (conditional + negative unconditional)."""
        # Pop custom key — model.forward() doesn't expect it
        is_dropped = inputs.pop("is_image_dropped", None)

        loss_type = getattr(self.args, "loss_type", "standard")
        if loss_type == "cfg":
            return self.compute_cfg_loss(model, inputs, is_dropped, return_outputs=return_outputs, **kwargs)
        elif loss_type == "cfg_margin":
            return self.compute_cfg_margin_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        elif loss_type == "cfg_conf_reg":
            return self.compute_cfg_conf_reg_loss(model, inputs, is_dropped, return_outputs=return_outputs, **kwargs)

        return super(QwenSFTTrainer, self).compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )

                if merger_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        super()._save_checkpoint(model, trial)

        if not self.args.lora_enable:
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        non_lora = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters(),
            require_grad_only=True,
        )


        if self.args.should_save:
            torch.save(non_lora, os.path.join(output_dir, "non_lora_state_dict.bin"))
            self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs.pop("is_image_dropped", None)  # Remove custom key before model forward
        labels = inputs.get("labels") if "labels" in inputs else None

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else None
            logits = outputs.logits if hasattr(outputs, "logits") else None

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)

    def _extract_prompt_and_reference(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer
    ) -> tuple:
        """
        Extract prompt (question only) and reference (answer) from input_ids and labels.

        In SFT dataset, labels == IGNORE_INDEX for prompt tokens, and labels == token_id for answer tokens.

        Returns:
            prompt_ids: tensor of prompt token ids (question part only)
            reference_text: decoded answer text
        """
        # Find where labels are not IGNORE_INDEX (answer starts)
        label_mask = labels != IGNORE_INDEX

        if label_mask.any():
            answer_start_idx = label_mask.nonzero(as_tuple=True)[0][0].item()
        else:
            # No answer found, use full input as prompt
            answer_start_idx = len(input_ids)

        # Extract prompt (everything before answer)
        prompt_ids = input_ids[:answer_start_idx]

        # Extract reference answer
        answer_ids = labels[label_mask]
        reference_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

        return prompt_ids, reference_text

    def _prepare_generation_inputs(
        self,
        batch_prompt_ids: List[torch.Tensor],
        original_inputs: Dict[str, torch.Tensor],
        tokenizer,
        device
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for generation by padding prompts and including vision inputs.
        """
        batch_size = len(batch_prompt_ids)

        # Pad prompts to same length (left padding for generation)
        max_prompt_len = max(p.shape[0] for p in batch_prompt_ids)

        padded_prompts = torch.full(
            (batch_size, max_prompt_len),
            tokenizer.pad_token_id,
            dtype=batch_prompt_ids[0].dtype,
            device=device
        )
        attention_masks = torch.zeros(
            (batch_size, max_prompt_len),
            dtype=torch.long,
            device=device
        )

        # Right padding (Qwen uses right padding)
        for i, prompt in enumerate(batch_prompt_ids):
            prompt_len = len(prompt)
            padded_prompts[i, :prompt_len] = prompt
            attention_masks[i, :prompt_len] = 1

        gen_inputs = {
            "input_ids": padded_prompts,
            "attention_mask": attention_masks,
        }

        if "mm_token_type_ids" in original_inputs:
            padded_mm_token_type_ids = torch.zeros(
                (batch_size, max_prompt_len),
                dtype=original_inputs["mm_token_type_ids"].dtype,
                device=device,
            )
            for i, prompt in enumerate(batch_prompt_ids):
                prompt_len = len(prompt)
                padded_mm_token_type_ids[i, :prompt_len] = original_inputs["mm_token_type_ids"][i, :prompt_len]
            gen_inputs["mm_token_type_ids"] = padded_mm_token_type_ids

        # Add vision inputs if present
        if "pixel_values" in original_inputs:
            gen_inputs["pixel_values"] = original_inputs["pixel_values"]
        if "image_grid_thw" in original_inputs:
            gen_inputs["image_grid_thw"] = original_inputs["image_grid_thw"]
        if "pixel_values_videos" in original_inputs:
            gen_inputs["pixel_values_videos"] = original_inputs["pixel_values_videos"]
        if "video_grid_thw" in original_inputs:
            gen_inputs["video_grid_thw"] = original_inputs["video_grid_thw"]
        if "second_per_grid_ts" in original_inputs:
            gen_inputs["second_per_grid_ts"] = original_inputs["second_per_grid_ts"]

        return gen_inputs

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Override evaluation_loop to support generation-based evaluation.

        If compute_metrics is provided and prediction_loss_only is False,
        this method will use model.generate() to produce text outputs
        and pass them to compute_metrics as GenerativeEvalPrediction.

        Your compute_metrics function should accept either:
        - GenerativeEvalPrediction with .predictions (List[str]) and .references (List[str])
        - Or a dict with 'predictions' and 'references' keys
        """
        args = self.args

        # Determine if we should do generation-based evaluation
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # If no compute_metrics or loss_only, fall back to default behavior
        if prediction_loss_only or self.compute_metrics is None:
            return super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only,
                ignore_keys,
                metric_key_prefix
            )

        # Generation-based evaluation
        logger.info(f"\n***** Running {description} (Generation Mode) *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        logger.info(f"  Batch size = {self.args.eval_batch_size}")

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        # Get processor/tokenizer
        tokenizer = self.processing_class.tokenizer

        # Setup generation config
        generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=getattr(args, 'generation_max_new_tokens', 512),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Unwrap model for generation
        unwrapped_model = self.accelerator.unwrap_model(model)

        all_predictions = []
        all_references = []
        all_losses = []

        for step, inputs in enumerate(dataloader):
            # Move inputs to device
            inputs = self._prepare_inputs(inputs)
            inputs.pop("is_image_dropped", None)  # Remove custom key before model forward

            batch_input_ids = inputs["input_ids"]
            batch_labels = inputs["labels"]
            batch_size = batch_input_ids.shape[0]

            # Compute loss using forward pass (optional, for logging)
            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss.detach()
                    # Gather loss across processes
                    loss = self.accelerator.gather(loss.repeat(batch_size))
                    all_losses.append(loss.cpu())

            # Extract prompts and references for each item in batch
            batch_prompt_ids = []
            batch_references = []

            for i in range(batch_size):
                prompt_ids, reference_text = self._extract_prompt_and_reference(
                    batch_input_ids[i],
                    batch_labels[i],
                    tokenizer
                )
                batch_prompt_ids.append(prompt_ids)
                batch_references.append(reference_text)

            # Prepare generation inputs
            gen_inputs = self._prepare_generation_inputs(
                batch_prompt_ids,
                inputs,
                tokenizer,
                batch_input_ids.device
            )

            # Generate
            with torch.no_grad():
                generated_ids = unwrapped_model.generate(
                    **gen_inputs,
                    generation_config=generation_config,
                )

            # Decode generated tokens (excluding prompt)
            for i in range(batch_size):
                prompt_len = len(batch_prompt_ids[i])
                new_tokens = generated_ids[i][prompt_len:]
                pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                all_predictions.append(pred_text)

            all_references.extend(batch_references)

            # Log progress
            if step % 10 == 0:
                logger.info(f"  Eval step {step}/{len(dataloader)}")

        # Gather predictions across processes if distributed
        if self.args.world_size > 1:
            # For distributed evaluation, we need to gather all predictions
            all_predictions = self._gather_predictions(all_predictions)
            all_references = self._gather_predictions(all_references)

        # Compute metrics
        eval_prediction = GenerativeEvalPrediction(
            predictions=all_predictions,
            references=all_references
        )

        metrics = self.compute_metrics(eval_prediction)

        # Add loss to metrics if available
        if all_losses:
            avg_loss = torch.cat(all_losses).mean().item()
            metrics[f"{metric_key_prefix}_loss"] = avg_loss

        # Prefix all metrics
        metrics = {
            f"{metric_key_prefix}_{k}" if not k.startswith(metric_key_prefix) else k: v
            for k, v in metrics.items()
        }

        self.log(metrics)

        return EvalLoopOutput(
            predictions=all_predictions,
            label_ids=all_references,
            metrics=metrics,
            num_samples=len(all_predictions),
        )

    def _gather_predictions(self, predictions: List[str]) -> List[str]:
        """Gather string predictions across all processes."""
        import torch.distributed as dist

        if not dist.is_initialized():
            return predictions

        world_size = dist.get_world_size()

        # Gather all predictions to rank 0
        gathered = [None] * world_size
        dist.all_gather_object(gathered, predictions)

        # Flatten the list
        all_predictions = []
        for preds in gathered:
            all_predictions.extend(preds)

        return all_predictions
