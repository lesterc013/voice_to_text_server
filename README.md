# This is the WhisperServer that uses gRPC for transcription

This is the cli pyinstaller code to create an exe build. Note to change the code in the --add-data based on where the model and processors are
pyinstaller --onedir src/whisper_grpc_server.py --add-data "whisper_downloads/whisper_medium_model:whisper_downloads/whisper_medium_model" --add-data "whisper_downloads/whisper_medium_processor:whisper_downloads/whisper_medium_processor"

Run this in the cli to create the exe build with the fine tuned model and processor
pyinstaller --onedir src/main.py --name voice-to-text-server --add-data "whisper_fine_tuning/whisper_medium_model_AawMaster:whisper_fine_tuning/whisper_medium_model_AawMaster" --add-data "whisper_fine_tuning/whisper_medium_processor_AawMaster:whisper_fine_tuning/whisper_medium_processor_AawMaster" --recursive-copy-metadata torchcodec --recursive-copy-metadata number-parser --hidden-import number_parser

For future builds, run below instead of the ones on top. The ones below already handle inputting certain files and folders that need manual inserting. The above is clean build
pyinstaller --clean voice-to-text-server.spec
