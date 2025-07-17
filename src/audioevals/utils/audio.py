import base64
import fractions
import io
import os
import wave
from typing import Generator, Optional, Type

import numpy as np
import soxr
from av import AudioFrame, AudioResampler
from pydub import AudioSegment


def convert_audio_ndarray_type(
    ndarray: np.ndarray, output_dtype: Type[np.float32 | np.int16]
) -> np.ndarray:
    if ndarray.dtype == output_dtype:
        return ndarray

    if ndarray.dtype == np.int16 and output_dtype == np.float32:
        return ndarray.astype(np.float32) / 32768.0

    if ndarray.dtype == np.float32 and output_dtype == np.int16:
        return np.round(ndarray * 32768.0).astype(np.int16)

    raise ValueError(
        f"Cannot convert audio ndarray from {ndarray.dtype} to {output_dtype}. "
        "Supported conversions: int16 to float32 and float32 to int16."
    )


def resample_audio_ndarray(
    audio: np.ndarray, input_sample_rate: int, output_sample_rate: int
) -> np.ndarray:
    if input_sample_rate == output_sample_rate:
        return audio

    # use soxr for high-quality resampling (supports multiple dtypes)
    resampled_audio = soxr.resample(
        audio, input_sample_rate, output_sample_rate, quality="HQ"
    )

    return resampled_audio


class AudioData:
    def __init__(self, audio_ndarray: np.ndarray, sample_rate: int):
        self.audio_ndarray = audio_ndarray

        # if the audio_ndarray is 2D, we need to convert it to 1D
        if self.audio_ndarray.ndim == 2:
            self.audio_ndarray = self.audio_ndarray.reshape(-1)
        elif self.audio_ndarray.ndim > 2:
            raise ValueError("AudioData only supports 1D or 2D ndarrays")

        self.sample_rate = sample_rate

    def get_duration_ms(self) -> float:
        return self.audio_ndarray.shape[0] / self.sample_rate * 1000.0

    def resample(self, sample_rate: int) -> "AudioData":
        return AudioData(
            resample_audio_ndarray(self.audio_ndarray, self.sample_rate, sample_rate),
            sample_rate,
        )

    def get_1d_array(self, dtype: Type[np.int16 | np.float32]) -> np.ndarray:
        assert self.audio_ndarray.ndim == 1

        return convert_audio_ndarray_type(self.audio_ndarray, dtype)

    def get_base64_bytes(self, dtype: Type[np.int16 | np.float32] = np.int16) -> str:
        return base64.b64encode(self.get_1d_array(dtype).tobytes()).decode("utf-8")

    def get_wav_bytes(self, dtype: Type[np.int16 | np.float32] = np.int16) -> bytes:
        # Convert the audio data to a WAV format
        wav_bytes = io.BytesIO()
        with wave.open(wav_bytes, "wb") as wav_writer:
            wav_writer.setnchannels(1)
            wav_writer.setsampwidth(2 if dtype == np.int16 else 4)
            wav_writer.setframerate(self.sample_rate)
            wav_writer.writeframes(self.get_1d_array(dtype).tobytes())
            return wav_bytes.getvalue()

    def get_wav_file_object(self) -> io.BytesIO:
        wav_bytes = io.BytesIO()
        with wave.open(wav_bytes, "wb") as wav_writer:
            wav_writer.setnchannels(1)
            wav_writer.setsampwidth(2)
            wav_writer.setframerate(self.sample_rate)
            wav_writer.writeframes(self.get_1d_array(np.int16).tobytes())
            wav_bytes.seek(0)
            return wav_bytes

    def get_audio_frame(self) -> AudioFrame:
        frame = AudioFrame.from_ndarray(
            self.get_1d_array(np.int16).reshape(1, -1),
            layout="mono",
            format="s16",
        )
        frame.sample_rate = self.sample_rate
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        return frame

    def stream_chunks(self, duration_ms: int) -> Generator["AudioData", None, None]:
        chunk_size = int(self.sample_rate * duration_ms / 1000.0)
        for i in range(0, len(self.audio_ndarray), chunk_size):
            yield AudioData(self.audio_ndarray[i : i + chunk_size], self.sample_rate)

    @classmethod
    def from_bytes(
        cls, audio_bytes: bytes, sample_rate: int, dtype: Type[np.float32 | np.int16]
    ) -> "AudioData":
        audio_ndarray = np.frombuffer(audio_bytes, dtype=dtype)
        return cls(audio_ndarray, sample_rate)

    @classmethod
    def from_base64(
        cls, audio_base64: str, sample_rate: int, dtype: Type[np.int16 | np.float32]
    ) -> "AudioData":
        audio_bytes = base64.b64decode(audio_base64)
        return cls.from_bytes(audio_bytes, sample_rate, dtype)

    @classmethod
    def from_wav_file(cls, wav_file_path: str) -> "AudioData":
        assert wav_file_path.endswith(".wav")

        with wave.open(wav_file_path, "rb") as wav_reader:
            sample_rate = wav_reader.getframerate()
            channels = wav_reader.getnchannels()
            if channels > 1:
                raise ValueError("Only mono audio is supported")
            if wav_reader.getsampwidth() == 2:
                audio_array = np.frombuffer(
                    wav_reader.readframes(wav_reader.getnframes()), dtype=np.int16
                )
            elif wav_reader.getsampwidth() == 4:
                audio_array = np.frombuffer(
                    wav_reader.readframes(wav_reader.getnframes()), dtype=np.float32
                )
            else:
                raise ValueError("Only 16-bit or 32-bit audio is supported")
        return cls(audio_array, sample_rate)

    @classmethod
    def from_wav_bytes(cls, wav_bytes: bytes) -> "AudioData":
        wav_io = io.BytesIO(wav_bytes)
        with wave.open(wav_io, "rb") as wav_reader:
            sample_rate = wav_reader.getframerate()
            if wav_reader.getsampwidth() == 2:
                audio_array = np.frombuffer(
                    wav_reader.readframes(wav_reader.getnframes()), dtype=np.int16
                )
            elif wav_reader.getsampwidth() == 4:
                audio_array = np.frombuffer(
                    wav_reader.readframes(wav_reader.getnframes()), dtype=np.float32
                )
            else:
                raise ValueError("Only 16-bit or 32-bit audio is supported")
        return cls(audio_array, sample_rate)

    @classmethod
    def from_audio_frame(cls, audio_frame: AudioFrame) -> "AudioData":
        if audio_frame.layout.name == "mono":
            return cls(audio_frame.to_ndarray(), audio_frame.sample_rate)
        elif audio_frame.layout.name == "stereo":
            # convert stereo to mono by averaging the two channels

            resampler = AudioResampler(
                format="s16",
                layout="mono",
                rate=audio_frame.sample_rate,
                frame_size=audio_frame.samples,
            )
            mono_audio_frames = resampler.resample(audio_frame)
            mono_audio_ndarray = np.concatenate(
                [frame.to_ndarray().reshape(-1) for frame in mono_audio_frames]
            )
            return cls(mono_audio_ndarray, audio_frame.sample_rate)
        else:
            raise ValueError("Only mono or stereo audio is supported")

    @classmethod
    def from_list(cls, audio_data_list: list["AudioData"]) -> "AudioData":
        assert len(audio_data_list) > 0, "AudioData list must not be empty"
        assert all(
            isinstance(audio_data, AudioData) for audio_data in audio_data_list
        ), "All items in the list must be AudioData"
        assert all(
            audio_data.sample_rate == audio_data_list[0].sample_rate
            for audio_data in audio_data_list
        ), "All AudioData objects must have the same sample rate"
        audio_ndarray = np.concatenate(
            [audio_data.get_1d_array(np.int16) for audio_data in audio_data_list]
        )
        return cls(audio_ndarray, audio_data_list[0].sample_rate)

    def save_to_file(self, file_path: str, format: str = "wav"):
        if os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if format == "wav":
            with wave.open(file_path, "wb") as wav_writer:
                wav_writer.setnchannels(1)
                wav_writer.setsampwidth(2)
                wav_writer.setframerate(self.sample_rate)
                wav_writer.writeframes(self.get_1d_array(np.int16).tobytes())
        else:
            raise ValueError(f"Unsupported audio format: {format}")

    @classmethod
    def from_audiosegment(cls, audio_segment: AudioSegment) -> "AudioData":
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        return cls.from_wav_bytes(buffer.getvalue())

    @classmethod
    def merge_audio_data(
        cls,
        audio1_data: "AudioData",
        audio2_data: "AudioData",
        output_path: Optional[str] = None,
    ):
        # Convert bytes to AudioSegment
        audio1 = AudioSegment.from_wav(audio1_data.get_wav_file_object())
        audio2 = AudioSegment.from_wav(audio2_data.get_wav_file_object())

        # Merge side by side (stereo - left and right channels)
        # merged = AudioSegment.from_mono_audiosegments(audio1, audio2)

        # OR merge sequentially (one after another)
        # merged = audio1 + audio2

        # Make both audio segments the same length
        max_duration = max(len(audio1), len(audio2))

        # Pad shorter audio with silence
        if len(audio1) < max_duration:
            silence = AudioSegment.silent(duration=max_duration - len(audio1))
            audio1 = audio1 + silence

        if len(audio2) < max_duration:
            silence = AudioSegment.silent(duration=max_duration - len(audio2))
            audio2 = audio2 + silence

        # OR overlay them (play simultaneously, mixed)
        merged = audio1.overlay(audio2)

        if output_path:
            merged.export(output_path, format="wav")

        return cls.from_audiosegment(merged)
