# AudioEvals

A comprehensive tool for evaluating generated TTS (Text-to-Speech) audio datasets with multiple evaluation metrics.

## Evaluation Types

### WER (Word Error Rate)
Measures the accuracy of speech-to-text transcription by comparing generated audio against ground truth transcripts.

### AudioBox Aesthetics
Evaluates audio quality using AudioBox's aesthetic scoring system, providing metrics for:
- CE (Content Enjoyment)
- CU (Content Usefulness) 
- PC (Production Complexity)
- PQ (Production Quality)

### VAD (Voice Activity Detection) Silence
Detects unnaturally long silences in generated audio using Silero VAD with RMS analysis. Provides:
- Maximum silence duration per file
- Total duration analysis
- Silence-to-speech ratio calculations

## Using as a Library

### Installation

```bash
pip install audioevals
```

### Basic Usage

```python
import asyncio
from audioevals.evals import wer_eval, audiobox_eval, vad_eval
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

### Complete Example

```python
import asyncio
from audioevals.evals import wer_eval, audiobox_eval, vad_eval
from audioevals.utils.audio import AudioData

async def evaluate_audio_file(file_path, transcript):
    """Complete evaluation of an audio file"""
    
    # Load audio data once
    audio_data = AudioData.from_wav_file(file_path)
    
    # Run all evaluations
    wer_result = await wer_eval.run_audio_data(audio_data, transcript)
    audiobox_result = audiobox_eval.run_audio_data(audio_data)
    vad_result = vad_eval.run_audio_data(audio_data)
    
    return {
        'wer': wer_result,
        'audiobox': audiobox_result,
        'vad': vad_result
    }

# Usage
results = asyncio.run(evaluate_audio_file(
    "/path/to/audio.wav", 
    "Hello world, this is a test."
))

print(f"WER: {results['wer']['wer_score']:.2f}%")
print(f"AudioBox PQ: {results['audiobox']['PQ']:.2f}")
print(f"Max silence: {results['vad']['max_silence_duration']:.2f}s")
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
  "004.wav": "What is the difference between weather and climate, and how do scientists study and predict both? Please explain the factors that influence weather patterns and how climate change affects long-term weather trends.",
  "005.wav": "I'm so sad to hear that. I'm here for you. What can I do to help?",
  "006.wav": "She let out a sudden (laughs) at the joke.",
  "007.wav": "He breathed a long (sighs) of relief when the test ended.",
  "008.wav": "Uhh, I'm not sure what to say. hmm... I'm just, ugh, a little bit confused."
}
```

## CLI Usage

You can run evaluations on the dataset by running:

```bash
audioevals --dataset {folder_name}
```

The results will be printed to console as well as saved to `{folder_name}/results.json` for inspection via something like jupyter notebook.

### Running Specific Evaluations

By default, the tool will run all the available evaluations, like WER, AudioBox aesthetics, VAD Silence. But it's possible to run only a select few with the `--evals` flag:

```bash
audioevals --dataset {folder_name} --evals wer vad
```

Available options are: `wer`, `audiobox`, `vad`

## Output

Results are saved to `{folder_name}/results.json` and include:
- Metadata about the evaluation run
- Individual file results for each evaluation type
- Summary statistics and averages