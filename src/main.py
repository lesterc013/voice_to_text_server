import logging

# Configure logging at the very start
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    handlers=[
        logging.FileHandler("whisper_server_log.log", mode="a"),
        logging.StreamHandler(),
    ],
)

import whisper_grpc_server

if __name__ == "__main__":
    whisper_grpc_server.serve()
