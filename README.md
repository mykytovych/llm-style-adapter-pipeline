# LLM Style Adapter Pipeline (QLoRA)

An end-to-end **parameter-efficient fine-tuning (PEFT)** pipeline for adapting Large Language Models (LLMs) to specific writing styles, personas, and linguistic patterns.

Built with [Unsloth](https://github.com/unslothai/unsloth), [Hugging Face Transformers](https://github.com/huggingface/transformers), and **QLoRA**, this project aims to make fine-tuning 8B-parameter language models accessible on limited hardware, including a single Google Colab T4 GPU.

The pipeline covers the entire workflow, from dataset acquisition and preprocessing to model fine-tuning and text generation with a trained LoRA adapter.

## Features

- **Automated Dataset Preparation** — Download datasets from Hugging Face and convert them into a consistent ChatML-compatible JSONL format.
- **Memory-Efficient Fine-Tuning** — Use 4-bit quantization and QLoRA to reduce GPU memory requirements during training.
- **Unsloth Optimization** — Leverage optimized training implementations to improve performance and reduce memory overhead.
- **Modular Architecture** — Keep dataset downloading, preprocessing, training, and inference in separate, reusable scripts.
- **Customizable Writing Styles** — Adapt models to literary styles, personas, dialects, and other text-generation patterns using instruction-response examples.
- **Adapter-Based Inference** — Load a trained LoRA adapter alongside its compatible base model to generate text in the target style.
- **Google Colab Support** — Run experiments in a notebook using a compatible GPU runtime.

## Tech Stack

| Category | Technologies |
| --- | --- |
| Language | Python |
| Deep Learning | PyTorch |
| Model Ecosystem | Hugging Face Transformers, Datasets |
| Fine-Tuning | PEFT, QLoRA, TRL |
| Optimization | Unsloth, bitsandbytes |
| GPU Acceleration | CUDA |
| Environment | Google Colab, Linux, WSL |

## Project Structure

```text
.
├── data/
│   ├── raw/                       # Downloaded datasets (not tracked by Git)
│   └── processed/                 # Prepared JSONL datasets (not tracked by Git)
├── notebooks/
│   └── 01_test_pipeline.ipynb     # Google Colab pipeline notebook
├── scripts/
│   ├── download_hf_dataset.py     # Downloads datasets from Hugging Face
│   ├── prepare_dataset.py         # Converts raw data into ChatML-compatible JSONL
│   ├── train.py                   # Runs QLoRA fine-tuning
│   └── generate.py                # Generates text using a trained adapter
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

> **Note:** The directory structure above describes the intended project layout. The scripts, notebook, and configuration files must be present in the repository for the corresponding commands to work.

## Requirements

- Python 3.10 or a version supported by your selected dependency stack.
- A compatible NVIDIA GPU and CUDA environment for local GPU training.
- A Google Colab runtime with a T4 GPU for the intended low-cost cloud workflow.
- A Hugging Face account and access token if the selected dataset or model requires authentication.
- Git and pip.

**Hardware note:** A 4-bit quantized 8B model can fit within the memory budget of a 16 GB T4 GPU under suitable configurations, but the actual requirements depend on sequence length, batch size, model architecture, and training settings. GPU memory constraints may still require gradient accumulation, gradient checkpointing, or shorter sequences.

## Quick Start

The recommended starting point is the Google Colab notebook:

`notebooks/01_test_pipeline.ipynb`

Alternatively, follow the steps below to run the pipeline manually in a compatible Linux or Google Colab environment.

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your repository URL and directory name.

### 2. Install Dependencies

Install the project's declared dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For environments requiring a separate Unsloth installation, follow the current instructions in the [official Unsloth documentation](https://docs.unsloth.ai/).

> **Compatibility note:** Unsloth, PyTorch, Transformers, TRL, PEFT, and bitsandbytes versions must be compatible with one another and with the selected CUDA environment. Avoid blindly mixing dependency versions from different installation guides.

### 3. Download a Dataset

This example uses a dataset intended for converting modern English into Shakespearean-style English.

```bash
python scripts/download_hf_dataset.py \
  --dataset_name "harpreetsahota/modern-to-shakesperean-translation" \
  --prompt_col "modern" \
  --response_col "shakespearean" \
  --output_csv "data/raw/shakespeare.csv"
```

The download script is expected to retrieve the dataset, select the specified columns, and save the resulting data as a CSV file.

Make sure the dataset identifier and column names match the actual dataset schema before running the command.

### 4. Prepare the Dataset

Convert the downloaded data into ChatML-compatible JSONL format and provide a system prompt describing the target writing style.

```bash
python scripts/prepare_dataset.py \
  --input "data/raw/shakespeare.csv" \
  --output "data/processed/train.jsonl" \
  --system-prompt "Write in Early Modern English with poetic and dramatic flair, inspired by Shakespearean language."
```

Each training example should follow a consistent conversation structure. For example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Write in Early Modern English with poetic and dramatic flair."
    },
    {
      "role": "user",
      "content": "What do you think about modern technology?"
    },
    {
      "role": "assistant",
      "content": "What wondrous engines now doth mortal man devise?"
    }
  ]
}
```

Save each example as a separate JSON object on its own line in the JSONL file.

**Important:** The dataset preparation script and the training script must agree on the JSONL schema. If `train.py` expects a plain-text field rather than a `messages` array, adjust the preprocessing or training configuration accordingly.

### 5. Fine-Tune the Model

Run QLoRA fine-tuning with the prepared dataset:

```bash
python scripts/train.py \
  --dataset "data/processed/train.jsonl" \
  --max_steps 60
```

The example uses 60 training steps for a short pipeline test. This is intended to validate the workflow, not to guarantee a high-quality style adaptation.

Training duration and output quality depend on the dataset size, model configuration, sequence length, and hyperparameters.

After training, the LoRA adapter is expected to be saved in:

```text
adapter_model/
```

The actual output directory depends on the configuration implemented in `train.py`.

### 6. Generate Text

Use the trained adapter to generate text in the target writing style:

```bash
python scripts/generate.py \
  --adapter_path "adapter_model" \
  --system_prompt "Write in Early Modern English with poetic and dramatic flair, inspired by Shakespearean language." \
  --prompt "What is your opinion on artificial intelligence?"
```

The inference script should load the compatible base model and the trained adapter, apply the system prompt, and generate a response.

The adapter can be loaded alongside the base model without permanently merging the weights. If merged weights are required for deployment, that can be implemented as a separate export step.

## Workflow

```text
Hugging Face Dataset
        |
        v
Dataset Download
        |
        v
Data Preprocessing
        |
        v
ChatML-Compatible JSONL
        |
        v
Base Model + QLoRA
        |
        v
Trained LoRA Adapter
        |
        v
Text Generation
```

## Customization

The pipeline can be adapted to different writing styles and use cases by changing the dataset, system prompt, and base model.

Potential applications include:

- **Literary Style Transfer** — Generate text inspired by historical or literary writing styles.
- **Persona Adaptation** — Adapt response patterns to a fictional character or a defined conversational persona.
- **Dialect Adaptation** — Explore linguistic variation using representative, high-quality training examples.
- **Domain-Specific Writing** — Adapt tone and formatting for documentation, technical explanations, or other specialized text.

The quality of the resulting model depends heavily on dataset quality, consistency, diversity, and the compatibility of the training examples with the intended task.

## Limitations

- Fine-tuning does not guarantee perfect imitation of a target style.
- Small datasets and very short training runs may produce limited or inconsistent improvements.
- QLoRA reduces memory requirements but does not eliminate the need for compatible hardware and software.
- The base model's capabilities, limitations, and safety characteristics remain relevant after fine-tuning.
- Dataset licensing and usage restrictions must be checked before redistribution or commercial use.

## Troubleshooting

| Issue | Suggested Action |
| --- | --- |
| CUDA out-of-memory error | Reduce sequence length or batch size; enable gradient checkpointing if supported. |
| Dependency conflicts | Use a clean virtual environment or Colab runtime and install compatible package versions. |
| Dataset column errors | Inspect the dataset schema and verify the requested column names. |
| Invalid JSONL format | Validate that every line contains a valid JSON object matching the training script's expected schema. |
| Adapter loading errors | Verify that the base model, adapter configuration, and PEFT versions are compatible. |

## License

This project is licensed under the MIT License. 
