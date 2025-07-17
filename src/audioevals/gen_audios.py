import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))

from tts_evals.tts import TTS_PROVIDERS, TTSProvider


async def generate_audio_files(
    provider: TTS_PROVIDERS = TTS_PROVIDERS.CARTESIA,
    model: Optional[str] = None,
    voice: Optional[str] = None,
    speed: float = 1.0,
    emotion: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    # Load transcripts
    transcripts_file = Path(__file__).parent / "transcripts.json"
    with open(transcripts_file, "r") as f:
        transcripts: List[str] = json.load(f)

    # Set up output directory
    if output_dir is None:
        output_path = Path(__file__).parent / f"{provider.value}_evals" / "audios"
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    transcripts_dict = {}
    for i, transcript in enumerate(transcripts):
        file_index = i + 1
        audio_filename = f"{file_index:03d}.wav"
        transcripts_dict[audio_filename] = transcript

    transcripts_dest = output_path.parent / "transcripts.json"
    with open(transcripts_dest, "w") as f:
        json.dump(transcripts_dict, f, indent=2)

    # Save TTS parameters
    tts_config = {
        "provider": provider.value,
        "model": model,
        "voice": voice,
        "speed": speed,
        "emotion": emotion,
        "timestamp": datetime.now().isoformat(),
        "output_directory": str(output_path),
        "total_files": len(transcripts),
    }

    tts_config_file = output_path.parent / "tts.json"
    with open(tts_config_file, "w") as f:
        json.dump(tts_config, f, indent=2)

    # Initialize TTS provider
    tts_provider = TTSProvider(provider)

    print(f"Generating {len(transcripts)} audio files using {provider.upper()} TTS...")
    print(f"Output directory: {output_path}")

    success_count = 0
    failed_files = []

    try:
        for i, transcript in enumerate(transcripts):
            try:
                print(f"Processing {i + 1}/{len(transcripts)}: {transcript[:50]}...")

                # Generate audio
                (
                    audio_data,
                    ttft,
                    _total_time,
                    _price,
                ) = await tts_provider.call_provider(
                    provider=provider,
                    text=transcript,
                    model=model,
                    voice=voice,
                    speed=speed,
                    emotion=emotion,
                )

                if audio_data is None:
                    print(f"  ❌ Failed to generate audio for transcript {i + 1}")
                    failed_files.append(i + 1)
                    continue

                # Save audio file
                output_file = output_path / f"{i + 1:03d}.wav"
                with open(output_file, "wb") as f:
                    f.write(audio_data.get_wav_bytes())

                success_count += 1
                duration = audio_data.get_duration_ms() / 1000
                print(
                    f"  ✅ Generated {output_file.name} (duration: {duration:.2f}s, TTFT: {ttft:.3f}s)"
                )

            except Exception as e:
                print(f"  ❌ Error processing transcript {i + 1}: {str(e)}")
                failed_files.append(i + 1)
                continue

    finally:
        await tts_provider.close()

    # Print summary
    print(f"\n{'=' * 50}")
    print("Generation complete!")
    print(f"✅ Successfully generated: {success_count}/{len(transcripts)} files")
    if failed_files:
        print(f"❌ Failed files: {failed_files}")
    print(f"📁 Output directory: {output_path}")
    print("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate audio files from transcripts using TTS"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cartesia",
        choices=[p.value for p in TTS_PROVIDERS],
        help="TTS provider to use (default: cartesia)",
    )
    parser.add_argument(
        "--model", type=str, help="Model name (uses provider default if not specified)"
    )
    parser.add_argument(
        "--voice", type=str, help="Voice ID (uses provider default if not specified)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Speech speed (default: 1.0)"
    )
    parser.add_argument("--emotion", type=str, help="Emotion for speech")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: tts_evals/{provider}_evals/audios)",
    )

    args = parser.parse_args()

    provider = TTS_PROVIDERS(args.provider)

    asyncio.run(
        generate_audio_files(
            provider=provider,
            model=args.model,
            voice=args.voice,
            speed=args.speed,
            emotion=args.emotion,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
