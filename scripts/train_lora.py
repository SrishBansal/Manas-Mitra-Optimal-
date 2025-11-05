import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from peft.utils.other import fsdp_auto_wrap_policy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_and_prepare_dataset(
    file_path: Union[str, Path],
    tokenizer: AutoTokenizer,
    max_source_length: int = 384,
    max_target_length: int = 192,
    test_size: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Load and preprocess the dataset.
    
    Args:
        file_path: Path to the dataset file (JSONL format)
        tokenizer: Tokenizer to use for preprocessing
        max_source_length: Maximum length of the input sequence
        max_target_length: Maximum length of the target sequence
        test_size: Fraction of the dataset to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict containing train and test splits
    """
    try:
        # Load dataset
        dataset = load_dataset("json", data_files=str(file_path))["train"]
        
        # Split into train/test
        dataset = dataset.train_test_split(
            test_size=test_size,
            shuffle=True,
            seed=seed,
        )
        
        # Preprocessing function
        def preprocess_function(examples):
            model_inputs = tokenizer(
                examples["input"],
                max_length=max_source_length,
                truncation=True,
                padding="max_length",
            )
            
            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["output"],
                    max_length=max_target_length,
                    truncation=True,
                    padding="max_length",
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Apply preprocessing
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Running tokenizer on dataset",
        )
        
        return tokenized_datasets
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def get_device() -> torch.device:
    """Get the appropriate device for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    model_name: str = "google/flan-t5-base",
    train_file: Union[str, Path] = "data/processed/train.jsonl",
    output_dir: Union[str, Path] = "models/checkpoints",
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    num_train_epochs: int = 5,
    learning_rate: float = 3e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_source_length: int = 384,
    max_target_length: int = 192,
    logging_steps: int = 10,
    save_steps: int = 100,
    eval_steps: int = 100,
    save_total_limit: int = 3,
    seed: int = 42,
    fp16: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    merge_and_save: bool = False,
    final_model_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Train a FLAN-T5 model with LoRA fine-tuning.
    
    Args:
        model_name: Name or path of the pre-trained model
        train_file: Path to the training data file (JSONL format)
        output_dir: Directory to save checkpoints and logs
        per_device_train_batch_size: Batch size per GPU/TPU core/CPU for training
        per_device_eval_batch_size: Batch size per GPU/TPU core/CPU for evaluation
        gradient_accumulation_steps: Number of updates steps to accumulate before performing a backward/update pass
        num_train_epochs: Total number of training epochs to perform
        learning_rate: Initial learning rate for AdamW optimizer
        warmup_ratio: Ratio of training steps for warmup
        weight_decay: Weight decay for AdamW optimizer
        max_source_length: Maximum input sequence length after tokenization
        max_target_length: Maximum output sequence length after tokenization
        logging_steps: Log every X updates steps
        save_steps: Save checkpoint every X updates steps
        eval_steps: Run evaluation every X steps
        save_total_limit: Limit the total amount of checkpoints
        seed: Random seed for reproducibility
        fp16: Whether to use 16-bit (mixed) precision training
        lora_rank: Rank of the LoRA update matrices
        lora_alpha: Alpha parameter for LoRA scaling
        lora_dropout: Dropout probability for LoRA layers
        merge_and_save: Whether to merge LoRA weights with base model and save
        final_model_dir: Directory to save the final merged model (if merge_and_save is True)
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    try:
        # Load tokenizer and model
        logger.info(f"Loading tokenizer and model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Configure LoRA
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q", "k", "v", "o", "wi", "wo"],
        )
        
        # Convert model to use LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Load and preprocess dataset
        logger.info("Loading and preprocessing dataset...")
        tokenized_datasets = load_and_prepare_dataset(
            file_path=train_file,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            seed=seed,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            return_tensors="pt",
            padding=True,
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,
            logging_dir=str(output_dir / "logs"),
            logging_strategy="steps",
            logging_steps=logging_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            seed=seed,
            fp16=fp16 and device.type == "cuda",
            report_to=["tensorboard"],
            remove_unused_columns=True,
            predict_with_generate=True,
            generation_max_length=max_target_length,
            generation_num_beams=4,
        )
        
        # Initialize Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"] if "test" in tokenized_datasets else None,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save the final model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Log training summary
        metrics = train_result.metrics
        metrics["train_samples"] = len(tokenized_datasets["train"])
        
        if "test" in tokenized_datasets:
            metrics["eval_samples"] = len(tokenized_datasets["test"])
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        # Merge LoRA weights if requested
        if merge_and_save:
            if not final_model_dir:
                final_model_dir = output_dir / "final_model"
            else:
                final_model_dir = Path(final_model_dir)
            
            logger.info(f"Merging LoRA weights and saving to {final_model_dir}...")
            
            # Create directory if it doesn't exist
            final_model_dir.mkdir(parents=True, exist_ok=True)
            
            # Merge and save
            model = model.merge_and_unload()
            model.save_pretrained(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
            
            logger.info(f"Final model saved to {final_model_dir}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5 with LoRA for Manas Mitra")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base",
                       help="Name or path of the pre-trained model")
    parser.add_argument("--train_file", type=str, default="data/processed/train.jsonl",
                       help="Path to the training data file (JSONL format)")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints",
                       help="Directory to save checkpoints and logs")
    
    # Training hyperparameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                       help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                       help="Batch size per device for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                       help="Total number of training epochs to perform")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                       help="Initial learning rate for AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Ratio of training steps for warmup")
    
    # Sequence length
    parser.add_argument("--max_source_length", type=int, default=384,
                       help="Maximum input sequence length after tokenization")
    parser.add_argument("--max_target_length", type=int, default=192,
                       help="Maximum output sequence length after tokenization")
    
    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save checkpoint every X updates steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                       help="Run evaluation every X steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                       help="Limit the total amount of checkpoints")
    
    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="Rank of the LoRA update matrices")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="Dropout probability for LoRA layers")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--no_fp16", action="store_false", dest="fp16",
                       help="Disable 16-bit (mixed) precision training")
    parser.add_argument("--merge_and_save", action="store_true",
                       help="Merge LoRA weights with base model and save final model")
    parser.add_argument("--final_model_dir", type=str, default=None,
                       help="Directory to save the final merged model (if merge_and_save is True)")
    
    args = parser.parse_args()
    
    # Call the training function with the parsed arguments
    train(
        model_name=args.model_name,
        train_file=args.train_file,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        fp16=args.fp16,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        merge_and_save=args.merge_and_save,
        final_model_dir=args.final_model_dir,
    )

    # Training arguments compatible with current Transformers version
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=10,
        seed=args.seed,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=4,
        remove_unused_columns=True,
        optim="adamw_torch",
        # Platform-specific settings
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_bf16_supported(),
        tf32=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        load_best_model_at_end=False,  # Disabled as it requires evaluation setup
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save adapter weights
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.merge_and_save:
        # Reload base model and merge adapters
        base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        from peft import PeftModel

        merged = PeftModel.from_pretrained(base_model, args.output_dir)
        merged = merged.merge_and_unload()  # merge LoRA weights into base
        os.makedirs(args.final_dir, exist_ok=True)
        merged.save_pretrained(args.final_dir)
        tokenizer.save_pretrained(args.final_dir)
        print(f"Merged model saved to {args.final_dir}")
    else:
        print("Training complete. Adapter weights saved. Use --merge_and_save to create a merged model.")


if __name__ == "__main__":
    main()
