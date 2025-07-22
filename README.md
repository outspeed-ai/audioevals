# AudioEvals

Effective evaluations for Text-to-Speech (TTS) systems.

## Evaluation Types

### WER (Word Error Rate)
Measures the accuracy of speech-to-text transcription by comparing generated audio against ground truth transcripts.

**Output format:**
```json
{
    "wer_score": 5.2,
    "stt_transcript": "hello world this is a test",
    "words_per_second": 3.4
}
```

### AudioBox Aesthetics
Evaluates audio quality using AudioBox's aesthetic scoring system, providing metrics for:
- CE (Content Enjoyment)
- CU (Content Usefulness) 
- PC (Production Complexity)
- PQ (Production Quality)

**Output format:**
```json
{
    "CE": 0.85,
    "CU": 0.92,
    "PC": 0.78,
    "PQ": 0.88
}
```

### VAD (Voice Activity Detection) Silence
Detects unnaturally long silences in generated audio using Silero VAD with RMS analysis. Provides:
- Maximum silence duration per file
- Total duration analysis
- Silence-to-speech ratio calculations

**Output format:**
```json
{
    "total_duration": 12.45,
    "max_silence_duration": 0.64,
    "silence_to_speech_ratio": 0.12,
    "silence_ratio": 0.05
}
```

### Pitch Analysis
Analyzes pitch characteristics and stability to detect pitch-related artifacts in TTS output:
- Mean pitch and standard deviation
- Pitch stability using coefficient of variation
- Semitone jump detection for octave errors
- Pitch range and frequency values

**Output format:**
```json
{
    "mean_pitch": 193.1,
    "pitch_std": 55.9,
    "pitch_min": 120.5,
    "pitch_max": 280.3,
    "pitch_range": 159.8,
    "pitch_values": [0.0, 195.2, 198.1, 0.0, 190.5, ...],
    "frame_size_ms": 10.0,
    "pitch_stability": 0.29,
    "semitone_jumps": 27,
    "octave_errors": 5,
    "max_semitone_jump": 8.2
}
```

### NISQA (Naturalness Assessment)
Predicts Mean Opinion Score (MOS) for speech naturalness using the NISQA model:
- MOS score on 1-5 scale (higher is better)
- Quality assessment for TTS naturalness

**Output format:**
```json
{
    "mos_score": 4.44
}
```

## Using as a Library

### Installation

```bash
pip install audioevals
```

### Basic Usage

```python
import asyncio
from audioevals.evals import wer_eval, audiobox_eval, vad_eval, pitch_eval, nisqa_eval
from audioevals.utils.audio import AudioData

# Load audio data
audio_data = AudioData.from_wav_file("/path/to/audio.wav")
transcript = "Hello world, this is a test."
```

### WER Evaluation

```python
# Using file path
wer_result = await wer_eval.run_single_file("/path/to/audio.wav", transcript)
print(f"WER: {wer_result['wer_score']:.2f}%")
print(f"STT: {wer_result['stt_transcript']}")
print(f"Words Per Second: {wer_result['words_per_second']}")

# Using AudioData instance
wer_result = await wer_eval.run_audio_data(audio_data, transcript)
print(f"WER: {wer_result['wer_score']:.2f}%")
```

### AudioBox Aesthetics Evaluation

```python
# Using file path
audiobox_result = audiobox_eval.run_single_file("/path/to/audio.wav")
print(f"Content Enjoyment: {audiobox_result['CE']:.2f}")
print(f"Production Quality: {audiobox_result['PQ']:.2f}")

# Using AudioData instance
audiobox_result = audiobox_eval.run_audio_data(audio_data)
print(f"Content Enjoyment: {audiobox_result['CE']:.2f}")
```

### VAD Silence Evaluation

```python
# Using file path
vad_result = vad_eval.run_single_file("/path/to/audio.wav")
print(f"Max silence duration: {vad_result['max_silence_duration']:.2f}s")
print(f"Silence/Speech ratio: {vad_result['silence_to_speech_ratio']:.2f}")

# Using AudioData instance
vad_result = vad_eval.run_audio_data(audio_data)
print(f"Max silence duration: {vad_result['max_silence_duration']:.2f}s")
```

### Pitch Analysis Evaluation

```python
# Using file path
pitch_result = pitch_eval.run_single_file("/path/to/audio.wav")
print(f"Mean pitch: {pitch_result['mean_pitch']:.1f}Hz")
print(f"Stability: CV={pitch_result['pitch_stability']:.2f}")
print(f"Semitone jumps: {pitch_result['semitone_jumps']}")

# Using AudioData instance
pitch_result = pitch_eval.run_audio_data(audio_data)
print(f"Mean pitch: {pitch_result['mean_pitch']:.1f}Hz")
```

### NISQA Evaluation

```python
# Using file path
nisqa_result = nisqa_eval.run_single_file("/path/to/audio.wav")
print(f"MOS score: {nisqa_result['mos_score']:.2f}")

# Using AudioData instance
nisqa_result = nisqa_eval.run_audio_data(audio_data)
print(f"MOS score: {nisqa_result['mos_score']:.2f}")
```

### Complete Example

```python
import asyncio
from audioevals.evals import wer_eval, audiobox_eval, vad_eval, pitch_eval, nisqa_eval
from audioevals.utils.audio import AudioData

async def evaluate_audio_file(file_path, transcript):
    audio_data = AudioData.from_wav_file(file_path)
    
    # Run all evaluations
    wer_result = await wer_eval.run_audio_data(audio_data, transcript)
    audiobox_result = audiobox_eval.run_audio_data(audio_data)
    vad_result = vad_eval.run_audio_data(audio_data)
    pitch_result = pitch_eval.run_audio_data(audio_data)
    nisqa_result = nisqa_eval.run_audio_data(audio_data)
    
    return {
        'wer': wer_result,
        'audiobox': audiobox_result,
        'vad': vad_result,
        'pitch': pitch_result,
        'nisqa': nisqa_result
    }

# Usage
results = asyncio.run(evaluate_audio_file(
    "/path/to/audio.wav", 
    "Hello world, this is a test."
))
```

## Dataset Structure (CLI usage)

The audioevals CLI expects datasets to be structured in a folder, in the following way:

```
{folder_name}/
├── audios/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── transcripts.json
```

Where `transcripts.json` should be a map of audio file name to its ground truth transcript, such as:

```json
{
  "001.wav": "He shouted, 'Everyone, please gather 'round! Here's the plan: 1) Set-up at 9:15 a.m.; 2) Lunch at 12:00 p.m. (please RSVP!); 3) Playing — e.g., games, music, etc. — from 1:15 to 4:45; and 4) Clean-up at 5 p.m.'",
  "002.wav": "Hey! What's up? Don't be shy, what can I do for you, cutie?",
  "003.wav": "I'm so excited to see you! I've been waiting for this moment for so long!",
}
```

## CLI Usage

You can run evaluations on the dataset by running:

```bash
audioevals --dataset {folder_name}
```

The results will be printed to console as well as saved to `{folder_name}/results.json` for inspection via something like a Jupyter notebook.

### Running Specific Evaluations

By default, the tool will run all the available evaluations. But it's possible to run only a select few with the `--evals` flag:

```bash
audioevals --dataset {folder_name} --evals wer vad pitch nisqa
```

Available options are: `wer`, `audiobox`, `vad`, `pitch`, `nisqa`

### Single File Evaluation

You can also evaluate a single audio file without needing a dataset structure:

```bash
audioevals --file /path/to/audio.wav
audioevals --file /path/to/audio.wav --evals pitch nisqa
```

Note: WER evaluation is automatically skipped for single files since it requires ground truth transcripts.

## Output

Results are saved to `{folder_name}/results.json` and include:
- Metadata about the evaluation run
- Individual file results for each evaluation type
- Summary statistics and averages