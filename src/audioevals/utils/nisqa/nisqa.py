import os
from pathlib import Path
from typing import Optional

import librosa as lb
import numpy as np
import pandas as pd
import torch

from audioevals.utils.nisqa import NISQA_lib as NL
from audioevals.utils.nisqa.NISQA_model import nisqaModel


class Nisqa:
    loaded_model = None

    @classmethod
    def load_model(cls, pretrained_model_path: Optional[str] = None):
        if pretrained_model_path is None:
            pretrained_model_path = str(
                Path(__file__).parent / "weights" / "nisqa_tts.tar"
            )
        if not os.path.exists(pretrained_model_path):
            raise ValueError(f"Pretrained model not found at: {pretrained_model_path}")

        args = {
            "mode": "predict_file",
            "deg": "dummy_path",  # This will be overridden
            "pretrained_model": pretrained_model_path,
            "tr_bs_val": 1,
            "tr_num_workers": 0,
            "ms_channel": None,
        }

        # Initialize NISQA model
        nisqa = nisqaModel(args)

        # Run prediction
        if nisqa.args["tr_parallel"]:
            nisqa.model = torch.nn.DataParallel(nisqa.model)

        cls.loaded_model = nisqa

    @classmethod
    def run_audio_data(cls, audio_data, sample_rate=48000, pretrained_model=None):
        """
        Returns:
            float: Predicted MOS score (1-5 scale)

        Example:
            >>> import numpy as np
            >>> import nisqa
            >>>
            >>> # Load your audio data as numpy array
            >>> audio_data = np.random.randn(48000)  # 1 second of audio
            >>>
            >>> # Predict MOS
            >>> mos_score = nisqa.run_audio_data(audio_data, pretrained_model='weights/nisqa_tts.tar')
            >>> print(f"Predicted MOS: {mos_score:.2f}")
        """

        if cls.loaded_model is None:
            cls.load_model(pretrained_model)

        nisqa = cls.loaded_model

        spec = _get_melspec_from_audio(
            audio_data,
            sr=nisqa.args["ms_sr"] or 48000,
            n_fft=nisqa.args["ms_n_fft"],
            hop_length=nisqa.args["ms_hop_length"],
            win_length=nisqa.args["ms_win_length"],
            n_mels=nisqa.args["ms_n_mels"],
            fmax=nisqa.args["ms_fmax"],
        )

        df = pd.DataFrame([{"deg": "temp_audio"}])

        nisqa.ds_val = _NumpyAudioDataset(df=df, spec=spec, args=nisqa.args)

        if nisqa.args["dim"]:
            y_hat, _ = NL.predict_dim(
                nisqa.model,
                nisqa.ds_val,
                nisqa.args["tr_bs_val"],
                nisqa.dev,
                num_workers=nisqa.args["tr_num_workers"],
            )
            # Return MOS prediction (first dimension)
            return float(y_hat[0, 0])
        else:
            y_hat, _ = NL.predict_mos(
                nisqa.model,
                nisqa.ds_val,
                nisqa.args["tr_bs_val"],
                nisqa.dev,
                num_workers=nisqa.args["tr_num_workers"],
            )
            return float(y_hat[0, 0])


def _get_melspec_from_audio(
    audio_data,
    sr=48000,
    n_fft=1024,
    hop_length=0.01,
    win_length=0.02,
    n_mels=48,
    fmax=16000,
):
    hop_length = int(sr * hop_length)
    win_length = int(sr * win_length)

    S = lb.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        S=None,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=1.0,
        n_mels=n_mels,
        fmin=0.0,
        fmax=fmax,
        htk=False,
        norm="slaney",
    )

    # Convert to dB
    spec = lb.core.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)

    return spec


class _NumpyAudioDataset(torch.utils.data.Dataset):
    def __init__(self, df, spec, args):
        self.df = df
        self.spec = spec
        self.args = args
        self.seg_length = args["ms_seg_length"]
        self.seg_hop_length = args.get("ms_seg_hop_length", 1)
        self.max_length = args.get("ms_max_segments", None)
        self.dim = args.get("dim", False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Segment the spectrogram
        if self.seg_length is not None:
            x_spec_seg, n_wins = NL.segment_specs(
                "temp_audio",  # dummy file path for error messages
                self.spec,
                self.seg_length,
                self.seg_hop_length,
                self.max_length,
            )
        else:
            x_spec_seg = self.spec
            n_wins = self.spec.shape[1]
            if self.max_length is not None:
                x_padded = np.zeros((x_spec_seg.shape[0], self.max_length))
                x_padded[:, :n_wins] = x_spec_seg
                x_spec_seg = np.expand_dims(x_padded.transpose(1, 0), axis=(1, 3))
                if not torch.is_tensor(x_spec_seg):
                    x_spec_seg = torch.tensor(x_spec_seg, dtype=torch.float)

        # Create dummy MOS values (not used in prediction)
        if self.dim:
            y = np.full((5, 1), np.nan).reshape(-1).astype("float32")
        else:
            y = np.full(1, np.nan).reshape(-1).astype("float32")

        return x_spec_seg, y, (index, n_wins)
