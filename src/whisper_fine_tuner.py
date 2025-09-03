import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from jiwer import wer
from whisper_downloader import model_save_path, processor_save_path

# 1. Initialize processor and model
processor = WhisperProcessor.from_pretrained(
    processor_save_path, language="en", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained(model_save_path)
model.config.use_cache = False  # Required for gradient checkpointing

# Taken from the direct path reference, and some relative
train_manifest_path = r"D:\AAW_Master\WhisperFineTune\TrainTest\WhisperTrain.manifest"
test_manifest_path = r"D:\AAW_Master\WhisperFineTune\TrainTest\WhisperTest.manifest"
training_output_path = r"whisper_fine_tuning\training_outputs"
fine_tuned_model_save_path = r"whisper_fine_tuning\whisper_medium_model_AawMaster"
fine_tuned_processor_save_path = (
    r"whisper_fine_tuning\whisper_medium_processor_AawMaster"
)


# 2. Load and verify dataset
data = load_dataset(
    "json", data_files={"train": train_manifest_path, "test": test_manifest_path}
)

# Check sample data
print("Sample data:", data["train"][0])

# Cast audio column to 16kHz
data = data.cast_column("audio_filepath", Audio(sampling_rate=16000))


# 3. Compute WER metric function to use later
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Calculate WER
    wer_score = wer(label_str, pred_str)
    return {"wer": wer_score}


# 4. Preprocessing function to use later
def prepare_dataset(batch):
    features = []
    labels = []

    for audio, text in zip(batch["audio_filepath"], batch["text"]):
        if len(audio["array"]) == 0 or not text.strip():
            continue

        try:
            # Process audio
            inputs = processor(audio["array"], sampling_rate=16000, return_tensors="pt")
            features.append(inputs.input_features[0])

            # Process text
            label = processor(text=text.strip(), return_tensors="pt").input_ids[0]
            labels.append(label)

        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    return {"input_features": features, "labels": labels}


# 5. Process dataset
processed_data = data.map(
    prepare_dataset,
    remove_columns=data["train"].column_names,
    batched=True,
    batch_size=4,
)

# Remove empty samples
processed_data = processed_data.filter(lambda x: len(x["labels"]) > 0)


# 6. Custom collator
class WhisperDataCollator:
    def __call__(self, features):
        input_features = [torch.tensor(f["input_features"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        # Pad sequences
        input_features = torch.nn.utils.rnn.pad_sequence(
            input_features,
            batch_first=True,
            padding_value=processor.feature_extractor.padding_value,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id
        )

        return {
            "input_features": input_features,
            "labels": labels,
            "attention_mask": (
                input_features != processor.feature_extractor.padding_value
            ).float(),
        }


model.config.proj_out = None

# 7. Training configuration to put into the Seq2SeqTrainer (3-epoch quick test)
training_args = Seq2SeqTrainingArguments(
    output_dir=training_output_path,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    predict_with_generate=True,
    generation_max_length=225,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to="none",
    logging_steps=10,
)

# 8. Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_data["train"],
    eval_dataset=processed_data["test"],
    data_collator=WhisperDataCollator(),
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,  # WER computation
)

# 9. Start training
print("Starting training...")
trainer.train()

# 10. Save final model

processor.save_pretrained(fine_tuned_processor_save_path)
trainer.save_model(fine_tuned_model_save_path)
