import io
import wave
from pathlib import Path

from PIL import Image


def write_silent_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(44100)
        audio_file.writeframes(b"\x00\x00" * 4410)


def image_bytes(size: tuple[int, int] = (600, 600), mode: str = "RGB") -> bytes:
    output = io.BytesIO()
    with Image.new(mode, size) as image:
        image.save(output, format="PNG")
    return output.getvalue()
