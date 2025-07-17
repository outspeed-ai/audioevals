# AudioEvals

A comprehensive tool for evaluating generated TTS (Text-to-Speech) audio datasets with multiple evaluation metrics.

## Dataset Structure

The audioevals tool expects datasets to be structured in a folder, in the following way:

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

## Usage

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

## Output

Results are saved to `{folder_name}/results.json` and include:
- Metadata about the evaluation run
- Individual file results for each evaluation type
- Summary statistics and averages