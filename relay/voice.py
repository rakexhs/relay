"""Voicemail tickets: transcribe the audio with Whisper and feed the text into the normal path.

faster-whisper is used when installed (`pip install 'relay[voice]'`), falling back to
openai-whisper. The model loads lazily on the first voicemail, not at import or startup.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from relay.domain import Channel, Priority, TicketCreate
from relay.providers.base import MissingDependencyError


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...


class WhisperTranscriber:
    def __init__(self, model_size: str = "base") -> None:
        self.model_size = model_size
        self._model = None
        self._backend = ""

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_size, device="auto", compute_type="int8")
            self._backend = "faster-whisper"
            return
        except ImportError:
            pass
        try:
            import whisper

            self._model = whisper.load_model(self.model_size)
            self._backend = "openai-whisper"
        except ImportError as exc:
            raise MissingDependencyError("faster-whisper", "voice") from exc

    def transcribe(self, audio_path: Path) -> str:
        if self._model is None:
            self._load()
        if self._backend == "faster-whisper":
            segments, _info = self._model.transcribe(str(audio_path), vad_filter=True)
            return " ".join(segment.text.strip() for segment in segments).strip()
        result = self._model.transcribe(str(audio_path))
        return str(result.get("text", "")).strip()


def voicemail_ticket(
    transcript: str, customer_id: str, priority: Priority = Priority.NORMAL
) -> TicketCreate:
    text = re.sub(r"\s+", " ", transcript).strip()
    if not text:
        raise ValueError("the voicemail transcript is empty")
    first_sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    subject = first_sentence if len(first_sentence) <= 80 else first_sentence[:77] + "..."
    return TicketCreate(
        customer_id=customer_id,
        subject=f"Voicemail: {subject}",
        body=f"(Transcribed voicemail) {text}",
        priority=priority,
        channel=Channel.VOICEMAIL,
    )
