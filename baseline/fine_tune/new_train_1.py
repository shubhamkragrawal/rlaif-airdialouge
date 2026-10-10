"""
Flags to be used on VertexAI when starting training:
--bucket=rlaif-data-bucket-rlaif-project
--data_blob=airdialogue_sampled_5k.jsonl
--output_prefix=models/sft-mistral-baseline (change)
--epochs=1 (change)

"""
import os
import json
import argparse
import logging
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from google.cloud import storage
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

from utils import download_from_gcs, upload_dir_to_gcs

# setting logger object for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# formatting air-dialogue dataset to our needs! TODO: Need to revist the max_flights parameter again!!
def format_flight_kb(kb, max_flights=10):
    # Check first we data is there at all!
    # print(kb)
    if not kb:
        return "No flights available."
    
    lines = []
    for i, flight in enumerate(kb[:max_flights], 1):
        if not isinstance(flight, dict):
            continue
        # print(flight)

        # Standardizing the dict with proper key names!
        parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in flight.items()]
        lines.append(f"Flight {i}: " + " | ".join(parts))

    return "\n".join(lines) if lines else "No flights available."

def format_row(row):

    # kb_text basically is the doct-list of flight details
    kb_text  = format_flight_kb(row.get("kb", []))
    dialogue = row.get("dialogue", [])

    # Creating the to and fro dialogue!
    # If no diaglue we return empty string! No need to check further!
    if not dialogue:
        return ""
    
    # THE PROMPT! 
    # FIX: short, direct system prompt — verbose version was eating the full context window
    # before [/INST] ever appeared, causing the collator to find nothing
    system_block = (
        "You are a flight booking agent. "
        "ONLY book flights listed in the database below. "
        "Never invent flights, prices, airlines, or any details not listed. "
        "If no flight matches the customer's needs, say so clearly.\n\n"
        f"Available Flights:\n{kb_text}"
    )

    # Organizing turns! To and from of agent and customer!
    turns = []
    for i, entry in enumerate(dialogue):

        # At even turns/numbers its customer else agent! In case of str!
        if isinstance(entry, str):
            role = "customer" if i % 2 == 0 else "agent"
            text = entry.strip()
        # In case of dict, tries to fetch a key called role which directly tells us who's speaking!
        elif isinstance(entry, dict):
            role = entry.get("role", "customer" if i % 2 == 0 else "agent")
            text = entry.get("utterance", entry.get("text", "")).strip()
        else:
            continue
        if text:
            turns.append((role, text))

    if not turns:
        return ""

    # The main logic of arranging: Mistral specific! <s>[INST] Instruction [/INST] Model Answer </s> | TODO: Need to revist in case of model change!
    full_text = ""
    i = 0
    first = True
    while i < len(turns):
        # It opens the sequence with <s>. Drops in the flight database, adds the customer's first question,
        # and closes the instruction with [/INST].
        role, text = turns[i]
        if role == "customer":
            if first:
                full_text += f"<s>[INST] {system_block}\n\n{text} [/INST] "
                first = False
            else:
                full_text += f"[INST] {text} [/INST] "
            if i + 1 < len(turns) and turns[i + 1][0] == "agent":
                full_text += f"{turns[i + 1][1]} </s>"
                i += 2
            else:
                full_text += "</s>"
                i += 1
        elif role == "agent" and first:
            full_text += f"<s>[INST] {system_block}\n\n [/INST] {text} </s>"
            first = False
            i += 1
        else:
            i += 1

    return full_text.strip()

# building a 5k sample dataset for now to tune and test the baseline model.
def build_dataset(jsonl_path):
    raw, formatted = [], []

    # reading the .json passed!
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    logger.info(f"Loaded {len(raw)} raw rows")

    token_counts = []
    for row in raw:

        # This is where the main formatting of the org. data is happening!
        text = format_row(row)
        if text:
            formatted.append({"text": text})

    # Just to know if we skipped any rows while cleaning!
    skipped = len(raw) - len(formatted)

    if skipped:
        logger.warning(f"Skipped {skipped} rows (empty after formatting)")
    logger.info(f"Final dataset size: {len(formatted)} examples")

    # Need so be an object of Dataset and not just a plain list!
    return Dataset.from_list(formatted)


def main(args):

    # Check first GPU available or not!
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        raise RuntimeError("No GPU detected - aborting.")

    # Talking to Storage API. 
    local_data = "/tmp/train.jsonl"

    # Download that data in the tmp/ location of the store!
    download_from_gcs(args.bucket, args.data_blob, local_data)

    # Preprocessing stagw!!
    dataset = build_dataset(local_data)

    # This is where tokenizing happens: Like WordtoVec thingy we learned in ML!
    logger.info(f"Loading tokenizer from {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)

    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Check if the data isn't too long! Compress!
    logger.info("Checking token lengths on first 100 examples")
    lengths = []
    for ex in dataset.select(range(min(100, len(dataset)))):
        ids = tokenizer(ex["text"], truncation=False)["input_ids"]
        lengths.append(len(ids))

    # Settings management!
    lengths.sort()
    logger.info(f"Token length stats (first 100 examples):")
    logger.info(f"  min={lengths[0]}  median={lengths[len(lengths)//2]}  max={lengths[-1]}")
    over = sum(1 for l in lengths if l > 4096)
    logger.info(f"  Examples over 4096 tokens: {over}/100")

    # This is basically to fit the model and our data: Like instead of having higher precison we get it down to lower precision! 4bit
    # Need to look more into these parameters!
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading model {args.model_id} in 4-bit NF4")

    # Now we load the model. We load it from transformers module we imported! HuggingFace librbary! GPT style!
    # Can comment the GPU specifc thingy and use cpu while running locally: args.model_id, device_map="cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Low Rank Adaptation: Reduces the trainable parameters. If we use all, we will run OOM.
    # Instead of (4096 * 4096), i will train (4096 * r) matrix.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32, # Usually set to (2 * r). Liek a volume knob.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # The attention part! Tells what to pay attention to!
        lora_dropout=0.1, # Random 10% neurons dropout! Like normally we do to reduce overfitting!
        bias="none",
        task_type="CAUSAL_LM",
    )

    """ 
    Running sanity check on the data token size and if it matches to the training data.
    added this snipet to know early signs of training failure
    """

    # These ids are specific to MISTRAL coming from its tokenizer! TODO: Can make it configurable in case of different model used!
    instruction_template_ids = [28792, 16289, 28793]
    response_template_ids    = [733, 28748, 16289, 28793]

    logger.info(f"instruction_template tokens: {tokenizer.convert_ids_to_tokens(instruction_template_ids)}")
    logger.info(f"response_template tokens   : {tokenizer.convert_ids_to_tokens(response_template_ids)}")

    response_collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template_ids,
        response_template=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )

    logger.info("Running collator sanity check")
    _ids = tokenizer(
        dataset[0]["text"],
        return_tensors="pt",
        truncation=True,
        max_length=4096,        
    )["input_ids"][0]

    _batch   = response_collator([{"input_ids": _ids, "attention_mask": torch.ones_like(_ids)}])
    _visible = (_batch["labels"][0] != -100).sum().item()
    logger.info(f"Collator check: {_visible}/{len(_ids)} tokens visible to loss")

    if _visible == 0:
        logger.error(f"First 80 tokens: {tokenizer.convert_ids_to_tokens(_ids[:80])}")
        raise RuntimeError(
            "Collator masking ALL tokens. "
            "The system prompt is still too long or token IDs are wrong. "
        )
    logger.info(f"Collator OK - {_visible} response tokens visible to loss")

    output_dir = "/tmp/sft_output"
    os.makedirs(output_dir, exist_ok=True)

    # fine-tune config for Mistral 7B model
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=0.3,
        optim="adamw_torch", # used A100 Nvidia GPU
        fp16=True, # accomodating larger sample size
        bf16=False,
        max_seq_length=4096,            
        logging_steps=25,
        # checkpoint straegy to not lose progress
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        dataset_text_field="text",
        dataset_num_proc=2,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=sft_config,
        tokenizer=tokenizer,
        data_collator=response_collator,
    )
    
    logger.info("-" * 60)
    logger.info("Starting SFT training v4")
    logger.info(f"  Examples       : {len(dataset)}")
    logger.info(f"  Epochs         : {args.epochs}")
    logger.info(f"  Effective batch: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
    logger.info(f"  Learning rate  : {sft_config.learning_rate}")
    logger.info(f"  Max seq length : {sft_config.max_seq_length}")
    logger.info(f"  Loss masking   : response-only")
    logger.info("-" * 60)

    trainer.train()

    logger.info(f"Saving adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Uploading to gs://{args.bucket}/{args.output_prefix}")
    upload_dir_to_gcs(args.bucket, output_dir, args.output_prefix)
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket",        required=True)
    parser.add_argument("--data_blob",     required=True)
    parser.add_argument("--output_prefix", default="models/sft-mistral-v4")
    parser.add_argument("--model_id",      default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--epochs",        type=int, default=3)
    args = parser.parse_args()
    main(args)
