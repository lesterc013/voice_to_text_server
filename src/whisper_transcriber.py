# This class allows the server to instantiate and make the transcription

import numpy as np
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from whisper_downloader import WhisperDownloaderInformation

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Get the paths from the single source of truth: the downloader object
whisper_download_paths = WhisperDownloaderInformation()
model_save_path = whisper_download_paths.model_save_path
processor_save_path = whisper_download_paths.processor_save_path


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
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_save_path)
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(processor_save_path)

    def transcribe_from_wav(self, wav_file_path):
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
        return transcription[0]


if __name__ == "__main__":
    transcriber = WhisperTranscriber()
    transcription = transcriber.transcribe_from_wav(
        r"C:\Users\mtsec\Documents\Audacity\aaw_test_wav.wav"
    )
    print(transcription)
