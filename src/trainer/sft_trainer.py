import copy
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
        self._cfg_metric_buffer: Dict[str, List[torch.Tensor]] = {}

        # `cfg_margin` returns a per-sample hinge, not a token-normalised sum, so
        # it cannot honour `num_items_in_batch`. Declaring that the model does not
        # accept loss kwargs makes Trainer apply its own division by
        # `gradient_accumulation_steps`, which is the correct averaging for this
        # objective. The other CFG losses *do* normalise by `num_items_in_batch`
        # themselves, so they keep the default behaviour.
        if getattr(self.args, "loss_type", "standard") == "cfg_margin":
            self.model_accepts_loss_kwargs = False

    # ------------------------------------------------------------------
    # Classifier-Free Guidance losses
    # ------------------------------------------------------------------

    def _record_cfg(self, **metrics) -> None:
        """Buffer CFG scalars so they land in the next ``Trainer.log()`` call.

        These used to be ``print()``ed on every rank on every micro-step: at
        grad-accum 64 on 2 GPUs that is ~500 lines and 128 blocking ``.item()``
        syncs per optimizer step, and none of it reached wandb/tensorboard.
        Values are kept as detached tensors and only converted to floats at log
        time, so the per-step device sync is gone as well.
        """
        for key, value in metrics.items():
            self._cfg_metric_buffer.setdefault(key, []).append(value)

    def log(self, *args, **kwargs):
        """Merge buffered CFG metrics into the trainer's own log payload."""
        if self._cfg_metric_buffer:
            merged = {
                key: sum(float(v) for v in vals) / len(vals)
                for key, vals in self._cfg_metric_buffer.items()
            }
            self._cfg_metric_buffer = {}
            if args and isinstance(args[0], dict):
                args = ({**args[0], **merged},) + args[1:]
            elif isinstance(kwargs.get("logs"), dict):
                kwargs["logs"] = {**kwargs["logs"], **merged}
        return super().log(*args, **kwargs)

    @staticmethod
    def _cfg_token_stats(logits_row, labels_row, want_entropy: bool = False):
        """Token-level CE sum (and optionally entropy sum) for one sequence.

        Returns ``(loss_sum, num_tokens, entropy_sum)``, summed over supervised
        tokens only.

        Rows are masked *before* the softmax. Computing ``log_softmax`` over the
        full ``(seq_len, vocab)`` block and masking afterwards allocated a
        152k-wide distribution for every position -- for the entropy term that
        was two such tensors per sample, which is gigabytes at realistic
        sequence lengths. Masking first keeps it to the handful of supervised
        positions.

        Everything is upcast to fp32: accumulating a 152k-way log-softmax in
        bf16 across thousands of tokens loses several significant digits, and
        the plain-SFT path this is meant to be comparable against does not
        accumulate in bf16 either.
        """
        shift_labels = labels_row[1:]
        mask = shift_labels != IGNORE_INDEX
        num_tokens = mask.sum().to(torch.float32)

        zero = logits_row.new_zeros((), dtype=torch.float32)
        if not bool(mask.any()):
            return zero, num_tokens, zero

        sel_logits = logits_row[:-1][mask].float()      # (num_supervised, vocab)
        sel_labels = shift_labels[mask]

        log_probs = torch.log_softmax(sel_logits, dim=-1)
        loss_sum = -log_probs.gather(1, sel_labels.unsqueeze(1)).squeeze(1).sum()

        entropy_sum = zero
        if want_entropy:
            entropy_sum = -(log_probs.exp() * log_probs).sum(dim=-1).sum()

        return loss_sum, num_tokens, entropy_sum

    @staticmethod
    def _cfg_denominator(num_items_in_batch, fallback_tokens):
        """Token denominator matching what the plain-SFT path gets from HF.

        ``Qwen*ForConditionalGeneration.forward`` takes ``**kwargs``, so
        ``Trainer.model_accepts_loss_kwargs`` is True and Trainer therefore skips
        its own division by ``gradient_accumulation_steps``, expecting
        ``compute_loss`` to have normalised by ``num_items_in_batch`` instead.
        The first version of this file ignored the kwarg entirely, which left
        every CFG run scaled by grad-accum steps relative to the ``standard``
        baseline -- so identical ``--learning_rate`` values were not comparable
        across the two.
        """
        if num_items_in_batch is None:
            return fallback_tokens.clamp(min=1.0)
        if torch.is_tensor(num_items_in_batch):
            return num_items_in_batch.to(
                device=fallback_tokens.device, dtype=torch.float32
            ).clamp(min=1.0)
        return float(max(int(num_items_in_batch), 1))

    def compute_cfg_loss(self, model, inputs, is_dropped, return_outputs=False,
                         num_items_in_batch=None, **kwargs):
        """CFG dual loss: ``E[CE | image kept] - w * E[CE | image dropped]``.

        Both terms are token-weighted (matching the plain-SFT baseline) and
        divided by the probability of the branch that produced them, which makes
        the per-step estimator unbiased for the objective above.
        """
        weight = self.args.cfg_loss_weight
        drop_prob = self.args.cfg_drop_prob
        batch_size = is_dropped.size(0)

        # A single-sample micro-batch is homogeneous by construction, so the
        # model's own fused CE can be used and Liger never materialises logits.
        # The branch is keyed on batch size -- which is identical on every rank,
        # since the distributed samplers pad -- and NOT on whether this rank's
        # batch happens to be mixed. Branching on the latter let one rank run the
        # fused path (labels kept, lm_head weight read directly) while another ran
        # the unfused path (labels popped, lm_head called as a module), giving
        # ZeRO-3 mismatched all-gather sequences across ranks.
        if batch_size == 1:
            outputs = model(**inputs)
            num_tokens = (inputs["labels"][:, 1:] != IGNORE_INDEX).sum().to(torch.float32)
            denom = self._cfg_denominator(num_items_in_batch, num_tokens)
            # outputs.loss is the mean over supervised tokens; recover the sum so
            # it renormalises against `denom` exactly like the manual path below.
            loss_sum = outputs.loss.float() * num_tokens

            if bool(is_dropped[0]):
                loss = -weight * loss_sum / (denom * drop_prob)
                self._record_cfg(cfg_loss_uncond=outputs.loss.detach())
            else:
                loss = loss_sum / (denom * (1.0 - drop_prob))
                self._record_cfg(cfg_loss_cond=outputs.loss.detach())
            self._record_cfg(cfg_frac_dropped=is_dropped.float().mean().detach())
            return (loss, outputs) if return_outputs else loss

        # batch_size > 1: labels are popped on *every* rank, not only on ranks
        # that happen to hold a mixed batch, so the forward graph stays identical
        # across the process group.
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        stats = [self._cfg_token_stats(logits[i], labels[i]) for i in range(batch_size)]
        loss_sums = torch.stack([s[0] for s in stats])
        tok_counts = torch.stack([s[1] for s in stats])

        cond_mask = ~is_dropped
        sum_cond = (loss_sums * cond_mask).sum()
        sum_uncond = (loss_sums * is_dropped).sum()
        total_tokens = tok_counts.sum()
        denom = self._cfg_denominator(num_items_in_batch, total_tokens)

        # Each group is weighted by its share of the batch's tokens (implicit in
        # summing rather than averaging) and then by its branch probability. That
        # is what makes this agree with the batch_size == 1 case above.
        #
        # Dividing group *means* by (1-p) and p -- as the first version did --
        # double-corrects, because a group mean has already removed the
        # probability factor. At p=0.1, w=0.1 the intended
        #   L_cond - 0.1 * L_uncond
        # came out as
        #   1.11 * L_cond - 1.0 * L_uncond,
        # a 10x overweight on the unconditional term.
        loss = sum_cond / (denom * (1.0 - drop_prob)) \
            - weight * sum_uncond / (denom * drop_prob)

        tok_cond = (tok_counts * cond_mask).sum().clamp(min=1.0)
        tok_uncond = (tok_counts * is_dropped).sum().clamp(min=1.0)
        self._record_cfg(
            cfg_loss_cond=(sum_cond / tok_cond).detach(),
            cfg_loss_uncond=(sum_uncond / tok_uncond).detach(),
            cfg_frac_dropped=is_dropped.float().mean().detach(),
        )
        return (loss, outputs) if return_outputs else loss

    def compute_cfg_margin_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Hinge on the CFG log-ratio.

        ``max(0, margin - [log P(y|img,txt) - log P(y|dropped,txt)])``; with
        ``L = -log P`` averaged over supervised tokens that is
        ``max(0, margin - L_uncond + L_cond)``. Only the conditional term carries
        gradient -- the unconditional pass runs under ``no_grad`` and is detached.

        This is a per-sample hinge rather than a token-normalised sum, so it does
        not consume ``num_items_in_batch``; see ``__init__`` for how the
        grad-accum averaging is arranged instead.
        """
        margin = self.args.cfg_loss_margin
        batch_size = inputs["input_ids"].size(0)

        inputs_uncond = {
            key: (value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value))
            for key, value in inputs.items()
        }

        # Zero whichever visual stream this batch actually carries. Only
        # `pixel_values` was handled before, so a video batch ran an
        # unconditional pass byte-identical to the conditional one: the hinge
        # collapsed to clamp(margin, 0), a constant with zero gradient, and
        # training silently did nothing at all.
        visual_keys = [
            key for key in ("pixel_values", "pixel_values_videos") if key in inputs_uncond
        ]
        if not visual_keys:
            raise ValueError(
                "loss_type='cfg_margin' needs a visual input to drop, but this batch "
                "carries neither `pixel_values` nor `pixel_values_videos`. For a "
                "text-only sample the conditional and unconditional passes are "
                "identical, so the hinge is a constant with no gradient. Filter "
                "text-only samples out of the dataset, or use a different --loss_type."
            )
        for key in visual_keys:
            inputs_uncond[key] = torch.zeros_like(inputs_uncond[key])

        if batch_size == 1:
            # Both passes keep their labels, so Liger's fused CE stays active and
            # no logits are materialised on either side.
            with torch.no_grad():
                loss_uncond = model(**inputs_uncond).loss.float()
            outputs_cond = model(**inputs)
            loss_cond = outputs_cond.loss.float()
            loss = torch.clamp(margin - loss_uncond.detach() + loss_cond, min=0.0)

            self._record_cfg(
                cfg_loss_cond=loss_cond.detach(),
                cfg_loss_uncond=loss_uncond.detach(),
                cfg_margin_active=(loss > 0).float().detach(),
            )
            return (loss, outputs_cond) if return_outputs else loss

        # batch_size > 1. Labels must be popped from BOTH dicts: with
        # use_liger_kernel=True (the default) the fused path returns logits=None
        # whenever labels are present, so reading `outputs.logits` on the
        # unconditional pass raised `TypeError: 'NoneType' is not subscriptable`
        # for every batch_size > 1 run.
        labels = inputs.pop("labels")
        inputs_uncond.pop("labels", None)

        with torch.no_grad():
            logits_uncond = model(**inputs_uncond).logits
            uncond_stats = [
                self._cfg_token_stats(logits_uncond[i], labels[i])
                for i in range(batch_size)
            ]
            per_sample_uncond = torch.stack(
                [s[0] / s[1].clamp(min=1.0) for s in uncond_stats]
            )

        outputs_cond = model(**inputs)
        logits_cond = outputs_cond.logits
        cond_stats = [self._cfg_token_stats(logits_cond[i], labels[i]) for i in range(batch_size)]
        per_sample_cond = torch.stack([s[0] / s[1].clamp(min=1.0) for s in cond_stats])

        loss = torch.clamp(
            margin - per_sample_uncond.detach() + per_sample_cond, min=0.0
        ).mean()

        self._record_cfg(
            cfg_loss_cond=per_sample_cond.mean().detach(),
            cfg_loss_uncond=per_sample_uncond.mean().detach(),
            cfg_margin_active=(
                margin - per_sample_uncond + per_sample_cond > 0
            ).float().mean().detach(),
        )
        return (loss, outputs_cond) if return_outputs else loss

    def compute_cfg_conf_reg_loss(self, model, inputs, is_dropped, return_outputs=False,
                                  num_items_in_batch=None, **kwargs):
        """CFG dual loss plus an entropy bonus on the unconditional predictions.

        ``L_cond - w * L_uncond - reg * H_uncond``: the entropy term is
        *subtracted*, so minimising the loss maximises the entropy of the
        model's predictions when the image is withheld.
        """
        weight = self.args.cfg_loss_weight
        drop_prob = self.args.cfg_drop_prob
        reg_weight = self.args.cfg_reg_weight
        batch_size = is_dropped.size(0)

        # Labels are popped unconditionally. The entropy term needs the full
        # distribution, so there is no fused-CE shortcut to preserve here, and
        # popping on every rank keeps the forward graph identical across the
        # process group (a data-dependent forward branch would deadlock ZeRO-3).
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        dropped = is_dropped.tolist()
        stats = [
            # Entropy is only needed for the unconditional rows. Computing it for
            # every sample and discarding it, as the first version did, paid for
            # the largest allocation in this function twice over. Branching here
            # is safe: this is elementwise maths on already-gathered logits, not
            # a module forward, so it cannot desynchronise collectives.
            self._cfg_token_stats(logits[i], labels[i], want_entropy=dropped[i])
            for i in range(batch_size)
        ]
        loss_sums = torch.stack([s[0] for s in stats])
        tok_counts = torch.stack([s[1] for s in stats])
        entropy_sums = torch.stack([s[2] for s in stats])

        cond_mask = ~is_dropped
        sum_cond = (loss_sums * cond_mask).sum()
        sum_uncond = (loss_sums * is_dropped).sum()
        sum_entropy = (entropy_sums * is_dropped).sum()
        denom = self._cfg_denominator(num_items_in_batch, tok_counts.sum())

        # Same token-share weighting as compute_cfg_loss; see the comment there
        # for why dividing group means by (1-p)/p double-corrects.
        loss = (
            sum_cond / (denom * (1.0 - drop_prob))
            - weight * sum_uncond / (denom * drop_prob)
            - reg_weight * sum_entropy / (denom * drop_prob)
        )

        tok_cond = (tok_counts * cond_mask).sum().clamp(min=1.0)
        tok_uncond = (tok_counts * is_dropped).sum().clamp(min=1.0)
        self._record_cfg(
            cfg_loss_cond=(sum_cond / tok_cond).detach(),
            cfg_loss_uncond=(sum_uncond / tok_uncond).detach(),
            cfg_entropy_uncond=(sum_entropy / tok_uncond).detach(),
            cfg_frac_dropped=is_dropped.float().mean().detach(),
        )
        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Dispatch to the configured CFG objective, or fall back to plain SFT."""
        # Pop the custom key -- model.forward() does not expect it.
        is_dropped = inputs.pop("is_image_dropped", None)

        loss_type = getattr(self.args, "loss_type", "standard")

        # `cfg_margin` derives its own unconditional pass and never reads
        # `is_dropped`; the other CFG losses cannot run without it.
        needs_flag = loss_type in ("cfg", "cfg_conf_reg")
        if loss_type == "standard" or (needs_flag and is_dropped is None):
            return super(QwenSFTTrainer, self).compute_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )
        if loss_type == "cfg":
            return self.compute_cfg_loss(
                model, inputs, is_dropped, return_outputs=return_outputs, **kwargs
            )
        if loss_type == "cfg_margin":
            return self.compute_cfg_margin_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )
        if loss_type == "cfg_conf_reg":
            return self.compute_cfg_conf_reg_loss(
                model, inputs, is_dropped, return_outputs=return_outputs, **kwargs
            )
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

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
