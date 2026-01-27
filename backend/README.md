# Manas Mitra – PEFT LoRA Fine-Tuning (FLAN-T5)

Anonymous, multilingual chatbot providing supportive, non-clinical psychological first-aid for students. This repo fine-tunes `google/flan-t5-base` (or `flan-t5-small`) with PEFT LoRA.

## Features
- Instruction-tuned base (FLAN-T5) for supportive Q&A.
- LoRA (r=8, alpha=16, dropout=0.05) targeting attention projections ["q","k","v"].
- Trainer-based pipeline with JSONL dataset (`input` → `output`).
- Optional merge step to produce a standalone merged model.
 - Built-in keyword-based safety filter for crisis language (English/Hindi/Bengali) during inference.

## Project Structure
- `data/dataset.jsonl` – Demo dataset with paraphrased PHQ-9/GAD-7/GHQ-style items and small-talk.
- `scripts/train_lora.py` – Fine-tuning script with PEFT LoRA.
- `scripts/infer.py` – Simple inference script.
- `scripts/safety.py` – Lightweight crisis keyword detector that returns supportive helpline guidance.
- `scripts/build_dataset.py` – Utility to expand/generate dataset entries (~150+ examples).
- `scripts/chat_cli.py` – Simple multi-turn chat interface that prepends the system prompt and enforces safety.
- `config/system_prompt.txt` – System instruction file used by the chat CLI.
- `requirements.txt` – Dependencies.

## Setup
1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Note: `bitsandbytes` is pinned to Linux via environment markers. On macOS, training will use CPU or Apple MPS automatically.

## Build/Expand Dataset
Generate or refresh `data/dataset.jsonl` with multi-turn, inline-scoring examples (EN/HI/BN):
```bash
python scripts/build_dataset.py
```
You can edit `scripts/build_dataset.py` to customize questions, languages, or response templates.

## Dataset
- Format: JSON Lines (`.jsonl`) with fields:
  - `input`: student-friendly question or user message
  - `output`: empathetic, non-clinical bot response
- Update `data/dataset.jsonl` with 100–300 examples for a stronger demo. Include paraphrases and different severity-range hints.

## Training (LoRA)
Run with defaults (FLAN-T5 base, demo dataset, 3 epochs):
```bash
python scripts/train_lora.py \
  --model_name google/flan-t5-base \
  --train_file data/dataset.jsonl \
  --output_dir outputs/lora-manas-mitra \
  --per_device_train_batch_size=4 \
  --num_train_epochs=3 \
  --learning_rate=5e-5 \
  --save_steps=50
```
This saves the LoRA adapter weights and tokenizer to `outputs/lora-manas-mitra/`.

## Merge Adapters into Base
Produce a standalone merged model:
```bash
python scripts/train_lora.py \
  --model_name google/flan-t5-base \
  --train_file data/dataset.jsonl \
  --output_dir outputs/lora-manas-mitra \
  --merge_and_save \
  --final_dir outputs/merged-manas-mitra
```
The merged model is saved to `outputs/merged-manas-mitra/`.

## Inference
- Using merged model:
```bash
python scripts/infer.py --model_path outputs/merged-manas-mitra --prompt "Have you been enjoying your hobbies?"
```
- Using adapter + base:
```bash
python scripts/infer.py \
  --model_path outputs/lora-manas-mitra \
  --base_model google/flan-t5-base \
  --prompt "Have you been enjoying your hobbies?"
```
Safety messages can be localized via `--locale` (currently used as a hint only):
```bash
python scripts/infer.py --model_path outputs/merged-manas-mitra --prompt "मैं जीना नहीं चाहता" --locale hi-IN
```

## Chat CLI (Multi-turn, System Prompt, Safety)
Run an interactive chat that uses `config/system_prompt.txt`, formats the conversation as Student/Manas Mitra, includes inline scoring prompts, and applies the safety filter before generation.

- Using merged model:
```bash
python scripts/chat_cli.py --model_path outputs/merged-manas-mitra --locale en-IN
```
- Using adapter + base:
```bash
python scripts/chat_cli.py \
  --model_path outputs/lora-manas-mitra \
  --base_model google/flan-t5-base \
  --locale en-IN
```
Type `exit` to quit.

## Safety & Crisis Guidance
- `scripts/safety.py` implements a simple, keyword-based safety check for English/Hindi/Bengali. If a message appears to indicate self-harm or suicide risk, the system returns a supportive crisis message with India helplines instead of generating a model response.
- This is a basic safety net, not a clinical risk assessment. Always route imminent risk to emergency services and encourage professional support.

## Notes & Recommendations
- Start with `flan-t5-small` on limited hardware. Switch to `flan-t5-base` for better quality.
- Expand dataset with multilingual examples (e.g., Hindi, Bengali) and culturally relevant phrasing.
- Keep responses supportive and non-diagnostic. Avoid medical advice; encourage professional help if severe risk is disclosed.
- Consider adding evaluation prompts and simple rule-based checks for safety (e.g., crisis keywords → suggest helplines).

## Example Test
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

input_text = "Have you been enjoying your hobbies?"
tokenizer = AutoTokenizer.from_pretrained("outputs/merged-manas-mitra")
model = AutoModelForSeq2SeqLM.from_pretrained("outputs/merged-manas-mitra")
outputs = model.generate(**tokenizer(input_text, return_tensors="pt"), max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## License
For research and educational purposes. Ensure compliance with local regulations and ethical guidelines for mental health support tools.
# Manas-Mitra
# Manas-Mitra-Optimal-
