import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def main():
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,                   # Mistral, same as before
        quantization_config=bnb_config,  # Same quantization config as before
        device_map="auto",
        trust_remote_code=True,
    )

    # Load your fine-tuned model
    ft_model = PeftModel.from_pretrained(base_model, "mistralai-medqa-finetune/checkpoint-2600")

    # Save locally
    ft_model.save_pretrained("model")

    # Push to Hub
    # os.system(f"huggingface-cli repo create daparasyte/mistral-7B-medQA --type model")
    ft_model.push_to_hub("daparasyte/mistral-7B-medQA")


if __name__=='__main__':
    main()