import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

def main(args):
    """
    Main function to fine-tune a language model using Unsloth and QLoRA.
    """
    print(f"Loading model: {args.model_name}")
    
    # 1. Load Model in 4-bit quantization (saves massive amount of VRAM)
    max_seq_length = 2048 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=None,          # Auto-detect (usually float16 or bfloat16)
        load_in_4bit=True,   # QLoRA quantization
    )

    # 2. Apply LoRA adapters
    # We only train 1% of the model parameters, freezing the rest.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0, 
        bias="none",    
        use_gradient_checkpointing="unsloth", # Saves VRAM
        random_state=3407,
    )

    # 3. Setup Chat Template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml", # We used ChatML in our prepare_dataset.py
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    # 4. Load and format dataset
    print(f"Loading dataset from {args.dataset}")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 5. Setup Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False, # Can make training 5x faster for short sequences
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=args.max_steps, # Use num_train_epochs=1 for full training
            learning_rate=2e-4,
            fp16=not FastLanguageModel.is_bf16_supported(),
            bf16=FastLanguageModel.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    # 6. Start Training
    print("Starting training...")
    trainer_stats = trainer.train()

    # 7. Save the LoRA adapter (the "Style")
    print(f"Saving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete! Pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QLoRA Fine-tuning pipeline")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit", help="HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, default="adapter_model", help="Where to save the LoRA weights")
    parser.add_argument("--max_steps", type=int, default=60, help="Number of training steps (for testing)")
    
    args = parser.parse_args()
    main(args)