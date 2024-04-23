import os
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from datasets import load_dataset
import torch
import torch.cuda as cuda
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import matplotlib.pyplot as plt
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import wandb
import transformers
from datetime import datetime

wandb.init(mode='disabled')
# wandb.login()


def tokenize(prompt, tokenizer, max_length):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt =f"""You are an expert web agent. Given that the user's objective and information about the last page visited, with observation, provide an accurate action to choose next.

    ### Information:
    {data_point['X']}

    ### Action:
    {data_point["X"]}
    """
    return tokenize(full_prompt, tokenizer, max_length=1024)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.savefig('plot.png')


def main():
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)


    # Load Dataset (using MedQA for this..)
    dataset = load_dataset('daparasyte/webdata', split='train')
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset['train']
    test_eval = dataset['test'].train_test_split(test_size=0.01) # just want a few test examples
    eval_dataset = test_eval['train']
    test_dataset = test_eval['test']
    hline = '-----------------------------------------'
    print(f'{hline}\nDataset\n{hline}\n')
    print(f'Train {train_dataset}\n')
    print(f'Val {eval_dataset}\n')
    print(f'Test {test_dataset}\n')

    print(f'First Test Sample:\n{test_dataset[0]}\n\n')


    # Load Base Model
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

    max_length = 1024 # This was an appropriate max length for my dataset

    # redefine the tokenize function and tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,  
        add_bos_token=True,  
    )
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_train_dataset = train_dataset.map(lambda x: generate_and_tokenize_prompt(data_point=x, tokenizer=tokenizer))
    tokenized_val_dataset = eval_dataset.map(lambda x: generate_and_tokenize_prompt(data_point=x, tokenizer=tokenizer))

    # print(f'Tokenized Train Sample:\n\n{tokenized_train_dataset[4]['input_ids']}\n')

    # untokenized_text = tokenizer.decode(tokenized_train_dataset[1]['input_ids']) 
    # print(f'Untokenized Text:\n\n{untokenized_text}\n')

    # print("Question: " + test_dataset[2]['Question'] + "\n")
    # print("Answer: " + test_dataset[2]['Answer'] + "\n")


    eval_prompt = f"""You are an expert web agent. Given that the user's objective and information about the last page visited, with observation, provide an accurate action to choose next.

    ### Information:
    {test_dataset[2]['X']}

    ### Action:
    """
    print(f'{hline}\nEval Prompt\n{hline}\n\n{eval_prompt}\n')

    # Apply the accelerator. You can comment this out to remove the accelerator.
    model = accelerator.prepare_model(model)
    # Re-init the tokenizer so it doesn't add padding or eos token
    eval_tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        add_bos_token=True,
    )
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"{hline}\nDevice Details\n{hline}\nDevice Name --> {cuda.get_device_properties(device).name}")
        print(f"Major:\t{cuda.get_device_properties(device).major}\nMinor:\t{cuda.get_device_properties(device).minor}")
        print(f"Memory Allocated:\t{cuda.get_device_properties(device).total_memory / 2**20} MB\n")
    else:
        device = torch.device('cpu')
        print(f"{hline}\nDevice Details\n{hline}\nDevice --> {device}\n\nMight want to consider switching to cuda\n")
        
    model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        print(f'{hline}\nBase model output\n{hline}\n\n{eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=300)[0], skip_special_tokens=True)}\n\n')


    # LoRA Setup
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    print(f'{hline}\nOriginal Model\n{hline}\n\n{model}\n\n')

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    # Apply the accelerator. You can comment this out to remove the accelerator.
    model = accelerator.prepare_model(model)

    print(f'\n\n{hline}\nPEFT Model\n{hline}\n\n{model}\n\n')
    
    # Skipping wandb 

    # wandb_project = "biomistral-finetune-identifier"
    # if len(wandb_project) > 0:
    #     os.environ["WANDB_PROJECT"] = wandb_project

    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True

    project = "webdata-finetune"
    base_model_name = "mistralai"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name

    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=5,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            max_steps=3000,
            learning_rate=2.5e-5,        # Want about 10x smaller than the Mistral learning rate
            logging_steps=100,
            log_level="info",
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir="./logs",        # Directory for storing logs
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=100,              # Save checkpoints every 100 steps
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=100,              # Evaluate every 100 steps
            do_eval=True,                # Perform evaluation at the end of training
            # use_reentrant=True,
            # report_to="wandb",           # Comment this out if you don't want to use weights & baises
            # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()


if __name__=='__main__':
    main()