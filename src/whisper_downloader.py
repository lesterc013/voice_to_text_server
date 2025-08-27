# Run this script to save the AutoModel and AutoProcessor into a folder
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# FYI: Learner laptop specs of 5050 8GB VRAM upper limit is medium cos medium needs 5GB VRAM but large needs 10GB VRAM
model_id = "openai/whisper-medium"

save_processor_path = "../whisper_downloads/processors"
save_model_path = "../whisper_downloads/models"

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
print("Downloading model")
model.save_pretrained(save_model_path)

processor = AutoProcessor.from_pretrained(model_id)
print("Downloading processor")
processor.save_pretrained(save_processor_path)
