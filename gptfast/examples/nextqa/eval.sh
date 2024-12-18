#!/bin/bash

BASE_MODEL="[YOUR_ARIA_PATH]"
TOKENIZER=rhymes-ai/Aria
LORA=""
IMG_SIZE=490
SAVE_ROOT=./eval/nextqa__$(basename $BASE_MODEL)__$(basename $LORA)__$IMG_SIZE

CMD="python examples/nextqa/evaluation.py \
    --base_model_path $BASE_MODEL \
    --tokenizer_path $TOKENIZER \
    --save_root $SAVE_ROOT \
    --image_size $IMG_SIZE \
    --batch_size 8"

if [ -n "$LORA" ]; then
    CMD="$CMD --peft_model_path $LORA"
fi

time $CMD
