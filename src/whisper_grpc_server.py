"""
Run this to start the grpc whisper server.
Should make this into an exe so that can run it easily.
"""

import grpc
from concurrent import futures
import logging
from whisper_transcriber import WhisperTranscriber

# Contains the base classes
import voice_to_text_pb2_grpc

# Contains the message types
import voice_to_text_pb2

# Configure logging at the very start
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Create an instance of the WhisperTranscriber class: This is the main class that will do the transcribing
logger.info("Loading VTT model. Standby...")
whisper_transcriber = WhisperTranscriber()
logger.info("Model successfully loaded")


# This class is based on the gRPC framework requirements
class WhisperGrpcServer(voice_to_text_pb2_grpc.VoiceToTextHandlerServicer):
    def TranscribeVoice(self, request, context):
        result = whisper_transcriber.transcribe_from_wav(request.audio_file_path)

        return voice_to_text_pb2.VttResponse(transcription=result)


def serve():
    port = "50052"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    voice_to_text_pb2_grpc.add_VoiceToTextHandlerServicer_to_server(
        WhisperGrpcServer(), server
    )
    server.add_insecure_port("[::]:" + port)
    server.start()
    logger.info(f"VTT server ready for transcription, listening on port: {port}\n")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
