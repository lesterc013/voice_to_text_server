# This class allows the server to instantiate and make the transcription
# I added alot of logger.info because want to provide some feedback on those areas where there is some perceived stalling

import sys
import os
import numpy as np
from pydub import AudioSegment
import torch
import logging

logger = logging.getLogger(__name__)

logger.info("Importing ML tools..")
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


model_path_dev = "whisper_fine_tuning\whisper_medium_model_AawMaster"
processor_path_dev = "whisper_fine_tuning\whisper_medium_processor_AawMaster"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def get_correct_path(behind_path):
    if getattr(sys, "frozen", False):
        # For PyInstaller, files are in the same directory as the executable
        # or in the _MEIPASS folder during execution
        base_dir = (
            os.path.dirname(sys.executable)
            if hasattr(sys, "_MEIPASS")
            else getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        )
        full_path = os.path.join(base_dir, behind_path)

        # Also check in the _MEIPASS temporary directory (for onefile mode)
        if not os.path.exists(full_path) and hasattr(sys, "_MEIPASS"):
            meipass_path = os.path.join(sys._MEIPASS, behind_path)
            if os.path.exists(meipass_path):
                return meipass_path

        print("full_path from frozen: " + full_path)
        return full_path
    else:
        return behind_path


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
        model_path = get_correct_path(model_path_dev)
        processor_path = get_correct_path(processor_path_dev)
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
