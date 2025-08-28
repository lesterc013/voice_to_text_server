from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class VttRequest(_message.Message):
    __slots__ = ("audio_file_path",)
    AUDIO_FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    audio_file_path: str
    def __init__(self, audio_file_path: _Optional[str] = ...) -> None: ...

class VttResponse(_message.Message):
    __slots__ = ("transcription",)
    TRANSCRIPTION_FIELD_NUMBER: _ClassVar[int]
    transcription: str
    def __init__(self, transcription: _Optional[str] = ...) -> None: ...
