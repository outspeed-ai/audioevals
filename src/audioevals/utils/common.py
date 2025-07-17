import os
import re

import aiohttp
import jiwer

from audioevals.utils.audio import AudioData
from typing import Dict, List, Optional, Tuple


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"-", "", text)
    text = re.sub(r"(?<!\w)[^\w\s]|[^\w\s](?!\w)", "", text)
    text = " ".join(text.split())
    return text


def get_similarity_score(reference: str, hypothesis: str) -> float:
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)

    wer = jiwer.wer(normalized_reference, normalized_hypothesis)
    return wer * 100


async def request_deepgram_stt(
    audio_data: AudioData,
) -> Optional[Tuple[str, List[Dict]]]:
    try:
        query_params = {
            "model": "nova-3",
            "language": "en",
            "filler_words": "true",
            "smart_format": "true",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url="https://api.deepgram.com/v1/listen",
                params=query_params,
                headers={
                    "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}",
                    "Content-Type": "audio/wav",
                },
                data=audio_data.get_wav_file_object(),
            ) as response:
                json_response = await response.json()
                
                # Extract transcript and word-level timestamps
                alternative = json_response["results"]["channels"][0]["alternatives"][0]
                transcript = alternative["transcript"]
                word_timestamps = alternative.get("words", [])
                
                return transcript, word_timestamps
    except Exception as e:
        print(f"Error in STT inference (deepgram): {str(e)}")
        return None
