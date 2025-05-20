import os
from transformers import AutoModelForCausalLM

model_list = ["cot", "lima", "oasst1", "science"]
for domain in ["alpaca", "gsm8k", "truthfulqa", "wikidyk", "nlgraph"]:
    for i in range(4):
        model_list.append(f"{domain}_cluster_{i}")

for model_name in model_list:
    if os.path.exists(model_name):
        continue
    model = AutoModelForCausalLM.from_pretrained("bunsenfeng/ds_" + model_name)
    model.save_pretrained(model_name)