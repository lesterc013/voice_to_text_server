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

whisper_transcriber = WhisperTranscriber()


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
    print("WhisperServer started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
