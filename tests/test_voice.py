import sys

import pytest

from relay.domain import Channel
from relay.providers.base import MissingDependencyError
from relay.voice import WhisperTranscriber, voicemail_ticket


def test_voicemail_ticket_uses_first_sentence_as_subject():
    ticket = voicemail_ticket("  Hi there.   Our exports keep timing out. ", "C-12")
    assert ticket.subject == "Voicemail: Hi there."
    assert ticket.channel == Channel.VOICEMAIL
    assert ticket.body.endswith("Our exports keep timing out.")


def test_empty_transcript_is_rejected():
    with pytest.raises(ValueError):
        voicemail_ticket("   ", "C-12")


def test_missing_whisper_explains_the_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "whisper", None)
    with pytest.raises(MissingDependencyError, match=r"relay\[voice\]"):
        WhisperTranscriber("tiny").transcribe(tmp_path / "x.wav")
