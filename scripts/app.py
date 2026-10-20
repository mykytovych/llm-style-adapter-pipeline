import gradio as gr
import argparse
from unsloth import FastLanguageModel
import torch

# Global variables to hold model, tokenizer, and configuration
model = None
tokenizer = None
system_prompt = ""

def load_model(adapter_path: str, sys_prompt: str):
    """Loads the base model and LoRA adapter."""
    global model, tokenizer, system_prompt
    print(f"Loading model and adapter from: {adapter_path}...")
    system_prompt = sys_prompt
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    # Enable native Unsloth optimization for faster inference
    FastLanguageModel.for_inference(model)
    print("Model loaded successfully! Launching UI...")

def generate_response(message: str, history: list) -> str:
    """Generates a response using the model based on chat history."""
    global model, tokenizer, system_prompt
    
    # Format messages for ChatML
    messages = [{"role": "system", "content": system_prompt}]
    
    # Append previous chat history
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
        
    # Append the current user message
    messages.append({"role": "user", "content": message})
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.7,
        top_p=0.9
    )
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the latest assistant response
    assistant_marker = "assistant\n"
    if assistant_marker in decoded_output:
        clean_response = decoded_output.split(assistant_marker)[-1].strip()
    else:
        # Fallback extraction
        response_start = decoded_output.find(message) + len(message)
        clean_response = decoded_output[response_start:].strip()
        
    return clean_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Gradio Web UI for the fine-tuned model")
    parser.add_argument("--adapter_path", type=str, default="adapter_model", help="Path to saved LoRA adapter")
    parser.add_argument("--system_prompt", type=str, default="You are William Shakespeare. Answer in Early Modern English with poetic and dramatic flair.", help="System prompt to define persona")
    
    args = parser.parse_args()
    
    # Load model into VRAM before starting the server
    load_model(args.adapter_path, args.system_prompt)
    
    # Initialize the Gradio Chat Interface
    demo = gr.ChatInterface(
        fn=generate_response,
        title="🎭 LLM Style Adapter Chat",
        description=f"**Current Persona:** {args.system_prompt}\n\nType a message below to chat with the fine-tuned model.",
        examples=[
            "What is your opinion on artificial intelligence?", 
            "Tell me a joke about programmers.", 
            "How doth the weather look today?"
        ],
    )
    
    # Launch with share=True to generate a public link (crucial for Google Colab)
    demo.launch(share=True)