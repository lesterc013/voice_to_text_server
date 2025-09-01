# This class allows the server to instantiate and make the transcription
# I added alot of logger.info because want to provide some feedback on those areas where there is some perceived stalling

import sys
import logging
import numpy as np
from pydub import AudioSegment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

logger = logging.getLogger(__name__)

logger.info("Importing AI libraries...")
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Commented this out cos this only works for dev build
# Get the paths from the single source of truth: the downloader object
# whisper_download_paths = WhisperDownloaderInformation()
# model_save_path = whisper_download_paths.model_save_path
# processor_save_path = whisper_download_paths.processor_save_path


# Function that if its the pyinstaller one, then need to add a ./_internal before the dev relative
# This is because the from_pretrained only accepts relative paths
def get_correct_path(behind_path):
    if getattr(sys, "frozen", False):
        return "./_internal" + behind_path
    else:
        return "." + behind_path


# Abstract out this method to not overcrowd the WhisperTranscriber class
def generate_samples_from_wav(wav_file_path):
    audio = AudioSegment.from_file(wav_file_path)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return samples


# Instantiate this class to transcribe using whisper model
class WhisperTranscriber:
    # Constructor will instantiate the model and processor
    def __init__(self):
        # This sets the correct path - prepend ./internal to the relative path
        model_path = get_correct_path("/whisper_downloads/whisper_medium_model")
        processor_path = get_correct_path("/whisper_downloads/whisper_medium_processor")
        logger.info("Loading VTT model. Standby...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(processor_path)
        logger.info("Model successfully loaded")

    def transcribe_from_wav(self, wav_file_path):
        logger.info("Transcription request received, transcribing...")
        samples = generate_samples_from_wav(wav_file_path)
        inputs = self.processor(
            samples,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
            sampling_rate=16_000,
        )
        inputs = inputs.to(device, torch.float32)

        # transcribe audio to ids
        generated_ids = self.model.generate(
            **inputs,
            return_timestamps=True,
            language="en",
            condition_on_prev_tokens=True,
        )

        transcription = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        logger.info("Transcription completed")
        return transcription[0]


if __name__ == "__main__":
    transcriber = WhisperTranscriber()
    transcription = transcriber.transcribe_from_wav(
        r"C:\Users\mtsec\Documents\Audacity\aaw_test_wav.wav"
    )
    print(transcription)
