import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from tqdm import tqdm

# Load your fine-tuned model and processor
fine_tuned_model_save_path = r"whisper_fine_tuning\whisper_medium_model_AawMaster"
fine_tuned_processor_save_path = (
    r"whisper_fine_tuning\whisper_medium_processor_AawMaster"
)

processor = WhisperProcessor.from_pretrained(fine_tuned_processor_save_path)
model = WhisperForConditionalGeneration.from_pretrained(fine_tuned_model_save_path)
model.config.forced_decoder_ids = None  # Disable forced decoding

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load test dataset
test_manifest_path = r"D:\AAW_Master\WhisperFineTune\TrainTest\WhisperTest.manifest"
dataset = load_dataset("json", data_files={"test": test_manifest_path})
dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))


def compute_wer_on_dataset():
    all_transcriptions = []
    all_references = []

    print("Computing WER on test dataset...")

    for sample in tqdm(dataset["test"], desc="Processing audio files"):
        try:
            # Load audio
            audio = sample["audio_filepath"]["array"]
            sampling_rate = sample["audio_filepath"]["sampling_rate"]
            reference_text = sample["text"]

            # Process audio
            input_features = processor(
                audio, sampling_rate=sampling_rate, return_tensors="pt"
            ).input_features

            input_features = input_features.to(device)

            # Generate transcription
            with torch.no_grad():
                pred_ids = model.generate(input_features, max_length=225)

            # Decode prediction
            transcription = processor.batch_decode(pred_ids, skip_special_tokens=True)[
                0
            ]

            # Store results
            all_transcriptions.append(transcription)
            all_references.append(reference_text)

        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    # Compute WER
    wer_score = wer(all_references, all_transcriptions)
    print(f"Word Error Rate (WER): {wer_score * 100:.2f}%")

    # Print some examples
    print("\nSample comparisons:")
    for i in range(min(5, len(all_references))):
        print(f"Reference: {all_references[i]}")
        print(f"Prediction: {all_transcriptions[i]}")
        print("---")

    return wer_score


if __name__ == "__main__":
    wer_score = compute_wer_on_dataset()
