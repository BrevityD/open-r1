NUM_GPUS=8
MODEL=open-r1/output/Llama-3.2-1B-Simple-RL

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.5"
TASK=aime24
OUTPUT_DIR=data/evals/Llama-3.2-1B-Instruct

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 