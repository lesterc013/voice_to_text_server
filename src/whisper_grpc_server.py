"""
Run this to start the grpc whisper server.
Should make this into an exe so that can run it easily.
"""

import signal
import sys
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
    handlers=[
        logging.FileHandler("whisper_server_log.log", mode="a"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# Create an instance of the WhisperTranscriber class: This is the main class that will do the transcribing
whisper_transcriber = WhisperTranscriber()


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
    logger.info(f"VTT server ready for transcription, listening on port: {port}")

    # Graceful shutdown handler
    def handle_shutdown(signum, frame):
        logger.info("Shutting down VTT server...")
        server.stop(0)  # 0 means stop in 0 seconds
        logger.info("VTT server stopped successfully.\n")
        sys.exit(0)  # Exit code 0 i.e. clean exit

    # Catch Ctrl+C or window close signals
    # Like an event handler: SIGINT is to catch Ctrl+C, SIGTERM is when close window or kill command
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
