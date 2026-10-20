import argparse
from unsloth import FastLanguageModel
import torch

def main(args):
    print(f"Loading model and adapter from: {args.adapter_path}...")
    max_seq_length = 2048
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True, 
        return_tensors="pt",
    ).to("cuda")

    print("\n--- Generating response ---\n")
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.7,
        top_p=0.9
    )
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response_start = decoded_output.find(args.prompt) + len(args.prompt)
    clean_response = decoded_output[response_start:].strip()
    
    print(clean_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using trained LoRA adapter")
    parser.add_argument("--adapter_path", type=str, default="adapter_model", help="Path to saved LoRA adapter")
    parser.add_argument("--system_prompt", type=str, required=True, help="System prompt used during training")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt to ask the model")
    
    args = parser.parse_args()
    main(args)