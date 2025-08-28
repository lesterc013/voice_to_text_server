"""
Run this file to test the grpc server
"""

from __future__ import print_function

import logging

import grpc
import voice_to_text_pb2
import voice_to_text_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    print("Try send audio file path")
    with grpc.insecure_channel("localhost:50052") as channel:
        stub = voice_to_text_pb2_grpc.VoiceToTextHandlerStub(channel)
        response = stub.TranscribeVoice(
            voice_to_text_pb2.VttRequest(
                audio_file_path=r"C:\Users\mtsec\Documents\Audacity\aaw_test_wav.wav"
            )
        )
    print("Transcription client received: " + response.transcription)


if __name__ == "__main__":
    logging.basicConfig()
    run()
