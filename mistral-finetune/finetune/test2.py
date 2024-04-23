from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
from peft import PeftModel

# base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    add_bos_token=True,
    trust_remote_code=True,
)

ft_model = PeftModel.from_pretrained(base_model, "mistralai-webdata-finetune/checkpoint-2300")

ft_model.save_pretrained("models/mistralai/Mistral-7B-Instruct-v0.2-aligned")

# model = AutoModelForCausalLM.from_pretrained("daparasyte/mistral-webagent")

# tokenizer = AutoTokenizer.from_pretrained(
#     "mistralai/Mistral-7B-Instruct-v0.2",
#     add_bos_token=True,
#     trust_remote_code=True,
# )

# device = 'cuda' 

# messages = [
#     {"role": "user", "content": "What is your favourite condiment?"},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     {"role": "user", "content": "Do you have mayonnaise recipes?"}
# ]

# encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

# model_inputs = encodeds.to(device)
# ft_model.to(device)

# generated_ids = ft_model.generate(model_inputs, max_new_tokens=128, do_sample=True)
# decoded = tokenizer.batch_decode(generated_ids)
# print(decoded[0])