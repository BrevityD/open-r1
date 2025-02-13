import pandas as pd
from vllm import LLM, SamplingParams

model_path = "pretrained_models/Llama-3.2-1B-Instruct"

df = pd.read_parquet("open-r1/data/aime_2024/data/train-00000-of-00001.parquet")

questions = df["problem"][:3].to_list()
answers = df['answer'][:3].to_list()

prompts = questions
sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=32768
    )
llm = LLM(
    model=model_path,
    max_model_len=32768,
    tensor_parallel_size=8,
    max_num_seqs=1,
    gpu_memory_utilization=0.5)

outputs = llm.generate(prompts, sampling_params)

for output, answer in zip(outputs, answers):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print("=**="*10)
    print(f">> Prompt:\n{prompt!r}\n>> Generated text:\n{generated_text!r}\n>> Answer:\n{answer!r}\n")