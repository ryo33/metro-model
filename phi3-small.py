import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "microsoft/Phi-3-small-8k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

messages = [
    {"role": "user", "content": "(I, have name, Tom)\n(I, see, a man (I, don't know, him))\n(I, want to do next, $)\n"},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 1000,
    "return_full_text": False,
    "temperature": 0.5,
    "do_sample": False,
}

start_time = time.time()
output = pipe(messages, **generation_args)
end_time = time.time()
print(output[0]['generated_text'])


elapsed_time = end_time - start_time
tokens_generated = sum(len(tokenizer.encode(text['generated_text'])) for text in output)
tokens_per_second = tokens_generated / elapsed_time

print(f"Tokens per second: {tokens_per_second}")
