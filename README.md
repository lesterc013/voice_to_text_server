# This is the WhisperServer that uses gRPC for transcription

This is the cli pyinstaller code to create an exe build. Note to change the code in the --add-data based on where the model and processors are
pyinstaller --onedir src/whisper_grpc_server.py --add-data "whisper_downloads/whisper_medium_model:whisper_downloads/whisper_medium_model" --add-data "whisper_downloads/whisper_medium_processor:whisper_downloads/whisper_medium_processor"

For adding fine-tuned whisper_medium
pyinstaller --onedir src/main.py --add-data "whisper_fine_tuning/whisper_medium_model_AawMaster:whisper_fine_tuning/whisper_medium_model_AawMaster" --add-data "whisper_fine_tuning/whisper_medium_processor_AawMaster:whisper_fine_tuning/whisper_medium_processor_AawMaster" --recursive-copy-metadata torchcodec
