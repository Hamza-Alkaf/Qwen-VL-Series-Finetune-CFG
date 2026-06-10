#!/bin/bash
loss_type="standard" # {standard, cfg, cfg_margin, cfg_conf_reg}
cfg_loss_margin=1.0 # For cfg_margin only
cfg_drop_prob=0.0 # For cfg, cfg_conf_reg
cfg_loss_weight=0.0 # For cfg, cfg_conf_reg
cfg_reg_weight=0.0 # For cfg_conf_reg

while [[ $# -gt 0 ]]; do
  case $1 in
    --loss_type)
      loss_type="$2"; shift 2 ;;
    --cfg_loss_margin)
      cfg_loss_margin="$2"; shift 2 ;;
    --cfg_drop_prob)
      cfg_drop_prob="$2"; shift 2 ;;
    --cfg_loss_weight)
      cfg_loss_weight="$2"; shift 2 ;;
    --cfg_reg_weight)
      cfg_reg_weight="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      exit 1 ;;
  esac
done

# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen3.5-4B"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=2
BATCH_PER_DEVICE=1
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))


# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# You should freeze the the merger also, becuase the merger is included in the vision_tower.

# If you want to set the min pixels and max pixels for Qwen3-VL, You should set as (N * 32 * 32)

# Please set the gradient_checkpointing to False when you are using LoRA with vision models.
# If you switch MODEL_NAME to a Qwen3.5 model, set `--disable_flash_attn2 True`.
# Flash Attention 2 raised CUDA errors for the Qwen3.5 series in local tests, so SDPA is the stable path for now.

deepspeed src/train/train_sft.py \
    --use_liger_kernel True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 1 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path textvqa_llava.json\
    --image_folder textvqa_images \
    --loss_type $loss_type \
    --cfg_loss_margin $cfg_loss_margin \
    --cfg_drop_prob $cfg_drop_prob \
    --cfg_loss_weight $cfg_loss_weight \
    --cfg_reg_weight $cfg_reg_weight \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir output/lora_vision_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4
