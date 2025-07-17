import os
from enum import Enum

import numpy as np
import onnxruntime

VAD_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(VAD_DIR_PATH, "silero_vad_v5_0.onnx")

class SileroVADModel:
    def __init__(self, sample_rate: int, path: str = MODEL_PATH):
        opts = onnxruntime.SessionOptions()

        # no parallelization since this is a small model and overhead from
        # managing multiple threads here is high
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        # only log fatal errors
        opts.log_severity_level = 4

        self.session = onnxruntime.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

        self._sample_rate = sample_rate

        if sample_rate not in [8000, 16000]:
            raise ValueError("Silero VAD only supports 8KHz and 16KHz sample rates")

        # window_size_samples: audio is processed in chunks of this size
        if sample_rate == 8000:
            self.window_size_samples = 256
            self._context_size = 32
        elif sample_rate == 16000:
            self.window_size_samples = 512
            self._context_size = 64

        self._sample_rate_nd = np.array(sample_rate, dtype=np.int64)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        self._rnn_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._input_buffer = np.zeros(
            (1, self._context_size + self.window_size_samples), dtype=np.float32
        )

    def _validate_input(self, x: np.ndarray):
        if np.ndim(x) == 1:
            x = np.expand_dims(x, 0)
        if np.ndim(x) > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.ndim}")

        if self._sample_rate / np.shape(x)[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x

    def __call__(self, x: np.ndarray) -> float:
        """Process a chunk of audio

        Args:
            x: Audio chunk (numpy array)
        """
        x = self._validate_input(x)

        if np.shape(x)[-1] != self.window_size_samples:
            raise ValueError(
                f"Provided number of samples is {np.shape(x)[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)"
            )

        x = np.concatenate((self._context, x), axis=1)

        ort_inputs = {"input": x, "state": self._rnn_state, "sr": self._sample_rate_nd}
        out, self._rnn_state = self.session.run(None, ort_inputs)

        self._context = x[..., -self._context_size :]

        return out.item()  # type: ignore

    def reset_states(self):
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        self._rnn_state = np.zeros((2, 1, 128), dtype=np.float32)


def calculate_rms(audio_samples: np.ndarray) -> float:
    """
    Calculate the RMS value of the audio samples.

    Args:
        audio_samples: audio samples to calculate RMS for

    Returns:
        RMS value
    """

    samples = audio_samples.astype(np.float32)
    if audio_samples.dtype != np.float32:
        samples /= 32768.0  # normalize 16‑bit PCM

    rms = np.sqrt(np.mean(samples**2))
    return rms.item()
