# Run this script to save the AutoModel and AutoProcessor into a folder

import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# FYI: Learner laptop specs of 5050 8GB VRAM upper limit is medium cos medium needs 5GB VRAM but large needs 10GB VRAM
model_id = "openai/whisper-medium"

processor_save_path = "./whisper_downloads/processors"
model_save_path = "./whisper_downloads/models"

# Only run the download if running directly. Else if importing as a module, then I only care about the class
if __name__ == "__main__":
    os.makedirs(processor_save_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    print("Downloading model")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    model.save_pretrained(model_save_path)

    print("Downloading processor")
    processor = AutoProcessor.from_pretrained(model_id)
    processor.save_pretrained(processor_save_path)


# To import in the transcriber so that we can have one source of truth for the model and processor paths
class WhisperDownloaderInformation:
    def __init__(self):
        self.model_save_path = model_save_path
        self.processor_save_path = processor_save_path
