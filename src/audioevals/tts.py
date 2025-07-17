# from dotenv import load_dotenv

# load_dotenv(override=True)
import asyncio
import json
import logging
import os
import time
from enum import Enum
from typing import Optional, Tuple

import aiohttp
import numpy as np
import pybase64
import requests
import websockets
from google import genai
from google.genai import types

from audioevals.utils.async_http_client import AsyncHttpClient, RequestParams
from audioevals.utils.audio import AudioData


class TTS_PROVIDERS(str, Enum):
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    DEEPGRAM = "deepgram"
    CARTESIA = "cartesia"
    BASETEN = "baseten"
    RIME = "rime"
    GEMINI = "gemini"
    PLAYHT = "playht"
    PLAYHT_TURBO = "playht_turbo"
    PLAYHT_GROQ = "playht_groq"
    PLAYHT_WEBSOCKET = "playht_websocket"
    RIME_WEBSOCKET = "rime_websocket"


PROVIDERS = {
    TTS_PROVIDERS.OPENAI: {
        "api_key": os.environ["OPENAI_API_KEY"],
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini-tts",
        "default_voice": "alloy",
        "fallback": TTS_PROVIDERS.ELEVENLABS,
        "price_per_1M_tokens_output": 12.0,
        "price_per_1M_tokens_input": 0.60,
    },
    TTS_PROVIDERS.ELEVENLABS: {
        "api_key": os.environ["ELEVENLABS_API_KEY"],
        "base_url": "https://api.elevenlabs.io/v1",
        "default_model": "eleven_turbo_v2_5",
        "default_voice": "MftN0gvsFPPOYnV3DU0Y",
        "fallback": TTS_PROVIDERS.OPENAI,
        "price_per_1M_tokens_input": 1320.0 / 22,
        "price_per_1M_tokens_output": 0.0,
    },
    TTS_PROVIDERS.DEEPGRAM: {
        "api_key": os.environ["DEEPGRAM_API_KEY"],
        "base_url": "https://api.deepgram.com/v1",
        "default_model": "aura-asteria-en",
        "default_voice": "aura-asteria-en",
        "fallback": TTS_PROVIDERS.OPENAI,
        "price_per_1M_tokens_input": 30.0,
        "price_per_1M_tokens_output": 0.0,
    },
    TTS_PROVIDERS.CARTESIA: {
        "api_key": os.environ["CARTESIA_API_KEY"],
        "base_url": "wss://api.cartesia.ai/tts/websocket",
        "default_model": "sonic-2",
        "default_voice": "bf0a246a-8642-498a-9950-80c35e9276b5",
        "fallback": TTS_PROVIDERS.OPENAI,
        "price_per_1M_tokens_input": 300.0 / 8,
        "price_per_1M_tokens_output": 0.0,
    },
    TTS_PROVIDERS.BASETEN: {
        "api_key": os.environ["BASETEN_API_KEY"],
        "base_url": "https://model-yqv07vjw.api.baseten.co/environments/production/predict",
        "default_model": "orpheus-2",
        "default_voice": "orpheus-2",
        "fallback": TTS_PROVIDERS.OPENAI,
    },
}


class TTSProvider:
    def __init__(self, provider: TTS_PROVIDERS = TTS_PROVIDERS.DEEPGRAM):
        assert provider in PROVIDERS, f"Invalid provider: {provider}"
        self.provider = provider
        self.logger = logging.getLogger(__name__)
        self.async_http_client = AsyncHttpClient(logger=self.logger)

    async def _request_deepgram(
        self,
        text: str,
        base_url: str,
        api_key: str,
        model: str,
        emotion: Optional[str],
    ) -> Tuple[Optional[AudioData], Optional[float], Optional[float], Optional[float]]:
        try:
            start_time = time.time()
            query_params = {
                "model": model,
                "container": "wav",
                "encoding": "linear16",
            }
            request_params = RequestParams(
                method="POST",
                endpoint="/speak",
                base_url=base_url,
                kwargs={
                    "headers": {
                        "Authorization": f"Token {api_key}",
                        "Content-Type": "application/json",
                    },
                    "json": {
                        "text": text,
                    },
                    "params": query_params,
                },
            )
            response = await self.async_http_client.request(request_params)
            buffer = b""
            first_chunk = True
            ttft = 0
            async for chunk in response.content:
                buffer += chunk
                if first_chunk:
                    ttft = time.time() - start_time
                    first_chunk = False
            total_time = time.time() - start_time
            audio_data = AudioData.from_wav_bytes(buffer)
            price = len(text) * (
                PROVIDERS[TTS_PROVIDERS.DEEPGRAM]["price_per_1M_tokens_input"] / 1000000
            )
            return audio_data, ttft, total_time, price
        except Exception as e:
            self.logger.error(f"Error in TTS inference (deepgram): {str(e)}")
            return None, None, None, None

    async def _request_openai(
        self,
        text: str,
        base_url: str,
        api_key: str,
        model: str,
        voice: str,
        speed: float,
        emotion: Optional[str],
    ) -> Tuple[Optional[AudioData], Optional[float], Optional[float], Optional[float]]:
        try:
            start_time = time.time()
            request_params = [
                RequestParams(
                    method="POST",
                    endpoint="/audio/speech",
                    base_url=base_url,
                    kwargs={
                        "headers": {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        "json": {
                            "input": text,
                            "model": model,
                            "voice": voice,
                            "response_format": "wav",
                            "speed": speed,
                        },
                    },
                ),
            ]

            response = await self.async_http_client.request(
                request_params=request_params
            )
            buffer = b""
            first_chunk = True
            ttft = 0
            async for chunk in response.content:
                buffer += chunk
                if first_chunk:
                    ttft = time.time() - start_time
                    first_chunk = False
            total_time = time.time() - start_time
            audio_data = AudioData.from_wav_bytes(buffer)

            price = None
            await asyncio.sleep(10)
            async with aiohttp.ClientSession() as new_session:
                async with new_session.get(
                    "https://api.openai.com/v1/organization/usage/completions",
                    headers={
                        "Authorization": f"Bearer {os.environ['OPENAI_ADMIN_API_KEY']}",
                    },
                    params={
                        "start_time": int(start_time) - 5,
                        "end_time": int(total_time + start_time) + 1000,
                        "limit": 1,
                        "models": [model],
                    },
                ) as resp:
                    json_data = await resp.json()
                    print(json_data)
                    results = json_data.get("data", [{}])[0].get("results", [{}])
                    if results:
                        price = results[0].get("input_tokens", 0) * (
                            PROVIDERS[TTS_PROVIDERS.OPENAI]["price_per_1M_tokens_input"]
                            / 1000000
                        )
                        price += results[0].get("output_audio_tokens", 0) * (
                            PROVIDERS[TTS_PROVIDERS.OPENAI][
                                "price_per_1M_tokens_output"
                            ]
                            / 1000000
                        )

            return audio_data, ttft, total_time, price
        except Exception as e:
            self.logger.error(f"Error in TTS inference (openai): {str(e)}")
            return None, None, None, None

    async def _request_elevenlabs(
        self,
        text: str,
        base_url: str,
        api_key: str,
        model: str,
        voice: str,
        speed: float,
        emotion: Optional[str],
    ) -> Tuple[Optional[AudioData], Optional[float], Optional[float], Optional[float]]:
        try:
            start_time = time.time()
            request_params = [
                RequestParams(
                    method="POST",
                    endpoint=f"/text-to-speech/{voice}/stream",
                    base_url=base_url,
                    kwargs={
                        "headers": {
                            "xi-api-key": api_key,
                            "Content-Type": "application/json",
                        },
                        "json": {
                            "text": text,
                            "model_id": model,
                            "voice_settings": {
                                "speed": speed,
                            },
                        },
                        "params": {
                            "output_format": "pcm_24000",
                            "optimize_streaming_latency": 2,
                        },
                    },
                ),
            ]

            response = await self.async_http_client.request(
                request_params=request_params
            )
            buffer = b""
            first_chunk = True
            ttft = 0
            async for chunk in response.content:
                buffer += chunk
                if first_chunk:
                    ttft = time.time() - start_time
                    first_chunk = False
            total_time = time.time() - start_time
            audio_data = AudioData.from_bytes(buffer, sample_rate=24000, dtype=np.int16)
            price = len(text) * (
                PROVIDERS[TTS_PROVIDERS.ELEVENLABS]["price_per_1M_tokens_input"]
                / 1000000
            )
            return audio_data, ttft, total_time, price
        except Exception as e:
            self.logger.error(f"Error in TTS inference (elevenlabs): {str(e)}")
            return None, None, None, None

    async def _request_cartesia(
        self,
        text: str,
        base_url: str,
        api_key: str,
        model: str,
        voice: str,
        speed: float,
        emotion: Optional[str],
    ) -> Tuple[Optional[AudioData], Optional[float], Optional[float], Optional[float]]:
        try:
            ws = await websockets.connect(
                base_url + f"/?api_key={api_key}&cartesia_version=2024-11-13",
            )
            start_time = time.time()

            experimental_controls = None
            if emotion:
                experimental_controls = {
                    "emotion": [emotion],
                }

            await ws.send(
                json.dumps(
                    {
                        "transcript": text,
                        "model_id": model,
                        "output_format": {
                            "container": "raw",
                            "encoding": "pcm_s16le",
                            "sample_rate": 24000,
                        },
                        "voice": {
                            "mode": "id",
                            "id": voice,
                            "__experimental_controls": experimental_controls,
                        },
                        "context_id": "123",
                    }
                )
            )
            buffer = b""
            first_chunk = True
            ttft = 0

            while True:
                message = await ws.recv()
                json_message = json.loads(message)
                if json_message.get("done", False):
                    break
                if json_message.get("data", None):
                    buffer += pybase64.b64decode(json_message["data"], validate=True)
                if first_chunk:
                    ttft = time.time() - start_time
                    first_chunk = False
            total_time = time.time() - start_time
            audio_data = AudioData.from_bytes(buffer, sample_rate=24000, dtype=np.int16)
            price = len(text) * (
                PROVIDERS[TTS_PROVIDERS.CARTESIA]["price_per_1M_tokens_input"] / 1000000
            )
            return audio_data, ttft, total_time, price
        except Exception as e:
            self.logger.error(f"Error in TTS inference (cartesia): {str(e)}")
            return None, None, None, None

    async def _request_baseten(
        self,
        text: str,
        base_url: str,
        api_key: str,
        emotion: Optional[str],
    ) -> Tuple[Optional[AudioData], Optional[float], Optional[float], Optional[float]]:
        try:
            start_time = time.time()
            response = requests.post(
                base_url,
                headers={
                    "Authorization": f"Api-Key {api_key}",
                },
                json={
                    "voice": "tara",
                    "prompt": text,
                    "max_tokens": 10000,
                },
                stream=True,
            )

            buffer = b""
            first_chunk = True
            ttft = 0
            for chunk in response.iter_content():
                buffer += chunk
                if first_chunk:
                    ttft = time.time() - start_time
                    first_chunk = False
            total_time = time.time() - start_time
            audio_data = AudioData.from_bytes(buffer, sample_rate=24000, dtype=np.int16)
            price = total_time * (0.06250 / 60)
            return audio_data, ttft, total_time, price
        except Exception as e:
            self.logger.error(f"Error in TTS inference (baseten): {str(e)}")
            return None, None, None, None

    async def _request_rime(
        self,
        text: str,
        base_url: str,
        api_key: str,
        model: str,
        voice: str,
        emotion: Optional[str],
    ) -> Tuple[Optional[AudioData], Optional[float], Optional[float], Optional[float]]:
        try:
            if emotion:
                text = f"<{emotion}> {text} </{emotion}>"

            start_time = time.time()
            request_params = [
                RequestParams(
                    method="POST",
                    endpoint="/rime-tts",
                    base_url=base_url,
                    kwargs={
                        "headers": {
                            "Accept": "audio/wav",
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        "json": {
                            "speaker": voice,
                            "text": text,
                            "modelId": model,
                            "repetition_penalty": 1.5,
                            "temperature": 0.5,
                            "top_p": 0.5,
                            "samplingRate": 24000,
                            "max_tokens": 1200,
                        },
                    },
                ),
            ]

            response = await self.async_http_client.request(
                request_params=request_params
            )
            buffer = b""
            first_chunk = True
            ttft = 0
            async for chunk in response.content:
                buffer += chunk
                if first_chunk:
                    ttft = time.time() - start_time
                    first_chunk = False
            total_time = time.time() - start_time
            audio_data = AudioData.from_bytes(buffer, sample_rate=24000, dtype=np.int16)
            price = len(text) * (
                PROVIDERS[TTS_PROVIDERS.RIME]["price_per_1M_tokens_input"] / 1000000
            )
            return audio_data, ttft, total_time, price
        except Exception as e:
            self.logger.error(f"Error in TTS inference (rime): {str(e)}")
            return None, None, None, None

    async def _request_gemini(
        self,
        text: str,
        base_url: str,
        api_key: str,
        model: str,
        voice: str,
        emotion: Optional[str],
    ) -> Tuple[Optional[AudioData], Optional[float], Optional[float], Optional[float]]:
        try:
            start_time = time.time()

            client = genai.Client(api_key=api_key)
            generate_content_config = types.GenerateContentConfig(
                temperature=1,
                response_modalities=[
                    "audio",
                ],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice,
                        )
                    )
                ),
            )

            buffer = b""
            ttft = 0
            first_chunk = True

            # Use streaming for TTS
            async for chunk in await client.aio.models.generate_content_stream(
                model=model,
                contents=text,
                config=generate_content_config,
            ):
                # Check if chunk has candidates
                if not hasattr(chunk, "candidates") or not chunk.candidates:
                    self.logger.debug("Chunk has no candidates, skipping")
                    continue

                # Process each candidate
                for candidate in chunk.candidates:
                    if not hasattr(candidate, "content") or not candidate.content:
                        self.logger.debug("Candidate has no content, skipping")
                        continue

                    if (
                        not hasattr(candidate.content, "parts")
                        or not candidate.content.parts
                    ):
                        self.logger.debug("Content has no parts, skipping")
                        continue

                    # Process each part
                    for part in candidate.content.parts:
                        # Check for inline_data attribute
                        if hasattr(part, "inline_data") and part.inline_data:
                            inline_data = part.inline_data
                            if (
                                hasattr(inline_data, "data")
                                and inline_data.data is not None
                            ):
                                if first_chunk:
                                    ttft = time.time() - start_time
                                    first_chunk = False
                                    self.logger.debug(
                                        f"First audio chunk received, TTFT: {ttft:.3f}s"
                                    )
                                buffer += inline_data.data
                                self.logger.debug(
                                    f"Added {len(inline_data.data)} bytes to buffer"
                                )

            total_time = time.time() - start_time
            self.logger.debug(f"Total audio buffer size: {len(buffer)} bytes")

            if not buffer:
                self.logger.error("No audio data received from Gemini")
                return None, None, None, None

            audio_data = AudioData.from_bytes(buffer, sample_rate=24000, dtype=np.int16)
            audio_tokens = client.models.count_tokens(
                model=model,
                contents=[
                    types.Part.from_bytes(
                        data=audio_data.get_wav_bytes(),
                        mime_type="audio/wav",
                    )
                ],
            )
            price = (
                len(text)
                * PROVIDERS[TTS_PROVIDERS.GEMINI]["price_per_1M_tokens_input"]
                / 1000000
            ) + audio_tokens.total_tokens * (
                PROVIDERS[TTS_PROVIDERS.GEMINI]["price_per_1M_tokens_input"] / 1000000
            )
            return audio_data, ttft, total_time, price
        except Exception as e:
            self.logger.error(f"Error in TTS inference (gemini): {str(e)}")
            return None, None, None, None

    async def _request_playht(
        self,
        text: str,
        base_url: str,
        api_key: str,
        model: str,
        voice: str,
        emotion: Optional[str],
    ) -> Tuple[Optional[AudioData], Optional[float], Optional[float], Optional[float]]:
        try:
            start_time = time.time()
            request_params = [
                RequestParams(
                    method="POST",
                    endpoint="/tts/stream",
                    base_url=base_url,
                    kwargs={
                        "headers": {
                            "Accept": "audio/wav",
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "accept": "*/*",
                            "X-User-Id": os.environ["PLAYHT_USER_ID"],
                        },
                        "json": {
                            "text": text,
                            "voice": voice,
                            "sample_rate": 24000,
                            "output_format": "wav",
                            "voice_engine": model,
                            "quality": "low",
                            "emotion": emotion if emotion else "null",
                            "style_guidance": 10,
                        },
                    },
                ),
            ]

            response = await self.async_http_client.request(
                request_params=request_params
            )
            buffer = b""
            first_chunk = True
            ttft = 0
            async for chunk in response.content:
                buffer += chunk
                if first_chunk:
                    ttft = time.time() - start_time
                    first_chunk = False
            total_time = time.time() - start_time
            audio_data = AudioData.from_wav_bytes(buffer)
            price = (
                len(text)
                * PROVIDERS[TTS_PROVIDERS.PLAYHT]["price_per_1M_tokens_input"]
                / 1000000
            )
            return audio_data, ttft, total_time, price
        except Exception as e:
            self.logger.error(
                f"Error in TTS inference (playht): {str(e)}", exc_info=True
            )
            return None, None, None, None

    async def _request_groq(
        self,
        text: str,
        base_url: str,
        api_key: str,
        model: str,
        voice: str,
        speed: float,
        emotion: Optional[str],
    ) -> Tuple[Optional[AudioData], Optional[float], Optional[float], Optional[float]]:
        try:
            start_time = time.time()
            request_params = [
                RequestParams(
                    method="POST",
                    endpoint="/audio/speech",
                    base_url=base_url,
                    kwargs={
                        "headers": {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        "json": {
                            "input": text,
                            "model": model,
                            "voice": voice,
                            "response_format": "wav",
                            "speed": speed,
                        },
                    },
                ),
            ]

            response = await self.async_http_client.request(
                request_params=request_params
            )
            buffer = b""
            first_chunk = True
            ttft = 0
            async for chunk in response.content:
                buffer += chunk
                if first_chunk:
                    ttft = time.time() - start_time
                    first_chunk = False
            total_time = time.time() - start_time
            audio_data = AudioData.from_wav_bytes(buffer)

            price = len(text) * (
                PROVIDERS[TTS_PROVIDERS.PLAYHT_GROQ]["price_per_1M_tokens_input"]
                / 1000000
            )

            return audio_data, ttft, total_time, price
        except Exception as e:
            self.logger.error(f"Error in TTS inference (groq): {str(e)}")
            return None, None, None, None

    async def _request_playht_websocket(
        self,
        text: str,
        base_url: str,
        api_key: str,
        model: str,
        voice: str,
        emotion: Optional[str],
    ) -> Tuple[Optional[AudioData], Optional[float], Optional[float], Optional[float]]:
        try:
            async with aiohttp.ClientSession() as new_session:
                async with new_session.post(
                    f"{base_url}/websocket-auth",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "X-User-Id": os.environ["PLAYHT_USER_ID"],
                    },
                ) as response:
                    response_json = await response.json()
            websocket_url = response_json["websocket_urls"]["PlayDialog"]

            ws = await websockets.connect(websocket_url)

            ttsCommand = {
                "text": text,
                "voice": "s3://voice-cloning-zero-shot/775ae416-49bb-4fb6-bd45-740f205d20a1/jennifersaad/manifest.json",
                "output_format": "wav",
            }

            start_time = time.time()
            await ws.send(json.dumps(ttsCommand))

            buffer = b""
            first_chunk = True
            ttft = 0

            while True:
                message = await ws.recv()
                if isinstance(message, bytes):
                    if first_chunk:
                        ttft = time.time() - start_time
                        first_chunk = False
                    buffer += message
                    continue
                json_message = json.loads(message)
                if json_message.get("type") == "end":
                    break
            await ws.close()

            total_time = time.time() - start_time
            audio_data = AudioData.from_wav_bytes(buffer)
            price = (
                len(text)
                * PROVIDERS[TTS_PROVIDERS.PLAYHT_WEBSOCKET]["price_per_1M_tokens_input"]
                / 1000000
            )
            return audio_data, ttft, total_time, price
        except Exception as e:
            self.logger.error(
                f"Error in TTS inference (playht_websocket): {str(e)}", exc_info=True
            )
            return None, None, None, None

    async def _request_rime_websocket(
        self,
        text: str,
        base_url: str,
        api_key: str,
        model: str,
        voice: str,
        emotion: Optional[str],
    ) -> Tuple[Optional[AudioData], Optional[float], Optional[float], Optional[float]]:
        try:
            websocket_url = f"{base_url}?speaker={voice}&modelId={model}&audioFormat=pcm&reduceLatency=true&samplingRate=24000"

            ws = await websockets.connect(
                websocket_url,
                additional_headers={
                    "Authorization": f"Bearer {api_key}",
                },
            )

            start_time = time.time()
            await ws.send(text)
            await ws.send("<EOS>")

            buffer = b""
            first_chunk = True
            ttft = 0

            while True:
                try:
                    audio = await ws.recv()
                except websockets.exceptions.ConnectionClosedOK:
                    break
                if first_chunk:
                    ttft = time.time() - start_time
                    first_chunk = False
                buffer += audio

            total_time = time.time() - start_time
            audio_data = AudioData.from_bytes(buffer, sample_rate=24000, dtype=np.int16)
            price = (
                len(text)
                * PROVIDERS[TTS_PROVIDERS.RIME_WEBSOCKET]["price_per_1M_tokens_input"]
                / 1000000
            )
            return audio_data, ttft, total_time, price
        except Exception as e:
            self.logger.error(
                f"Error in TTS inference (rime_websocket): {str(e)}", exc_info=True
            )
            return None, None, None, None

    async def call_provider(
        self,
        provider: TTS_PROVIDERS,
        text: str,
        model: Optional[str],
        voice: Optional[str],
        speed: float,
        emotion: Optional[str],
    ) -> Tuple[Optional[AudioData], Optional[float], Optional[float], Optional[float]]:
        if provider in [TTS_PROVIDERS.DEEPGRAM]:
            return await self._request_deepgram(
                text,
                PROVIDERS[provider]["base_url"],
                PROVIDERS[provider]["api_key"],
                model or PROVIDERS[provider]["default_model"],
                emotion,
            )
        elif provider in [TTS_PROVIDERS.OPENAI]:
            return await self._request_openai(
                text,
                PROVIDERS[provider]["base_url"],
                PROVIDERS[provider]["api_key"],
                model or PROVIDERS[provider]["default_model"],
                voice or PROVIDERS[provider]["default_voice"],
                speed,
                emotion,
            )
        elif provider in [TTS_PROVIDERS.ELEVENLABS]:
            return await self._request_elevenlabs(
                text,
                PROVIDERS[provider]["base_url"],
                PROVIDERS[provider]["api_key"],
                model or PROVIDERS[provider]["default_model"],
                voice or PROVIDERS[provider]["default_voice"],
                speed,
                emotion,
            )
        elif provider in [TTS_PROVIDERS.CARTESIA]:
            return await self._request_cartesia(
                text,
                PROVIDERS[provider]["base_url"],
                PROVIDERS[provider]["api_key"],
                model or PROVIDERS[provider]["default_model"],
                voice or PROVIDERS[provider]["default_voice"],
                speed,
                emotion,
            )
        elif provider in [TTS_PROVIDERS.BASETEN]:
            return await self._request_baseten(
                text,
                PROVIDERS[provider]["base_url"],
                PROVIDERS[provider]["api_key"],
                emotion,
            )
        elif provider in [TTS_PROVIDERS.RIME]:
            return await self._request_rime(
                text,
                PROVIDERS[provider]["base_url"],
                PROVIDERS[provider]["api_key"],
                model or PROVIDERS[provider]["default_model"],
                voice or PROVIDERS[provider]["default_voice"],
                emotion,
            )
        elif provider in [TTS_PROVIDERS.GEMINI]:
            return await self._request_gemini(
                text,
                PROVIDERS[provider]["base_url"],
                PROVIDERS[provider]["api_key"],
                model or PROVIDERS[provider]["default_model"],
                voice or PROVIDERS[provider]["default_voice"],
                emotion,
            )
        elif provider in [TTS_PROVIDERS.PLAYHT, TTS_PROVIDERS.PLAYHT_TURBO]:
            return await self._request_playht(
                text,
                PROVIDERS[provider]["base_url"],
                PROVIDERS[provider]["api_key"],
                model or PROVIDERS[provider]["default_model"],
                voice or PROVIDERS[provider]["default_voice"],
                emotion,
            )
        elif provider in [TTS_PROVIDERS.PLAYHT_GROQ]:
            return await self._request_groq(
                text,
                PROVIDERS[provider]["base_url"],
                PROVIDERS[provider]["api_key"],
                model or PROVIDERS[provider]["default_model"],
                voice or PROVIDERS[provider]["default_voice"],
                speed,
                emotion,
            )
        elif provider in [TTS_PROVIDERS.PLAYHT_WEBSOCKET]:
            return await self._request_playht_websocket(
                text,
                PROVIDERS[provider]["base_url"],
                PROVIDERS[provider]["api_key"],
                model or PROVIDERS[provider]["default_model"],
                voice or PROVIDERS[provider]["default_voice"],
                emotion,
            )
        elif provider in [TTS_PROVIDERS.RIME_WEBSOCKET]:
            return await self._request_rime_websocket(
                text,
                PROVIDERS[provider]["base_url"],
                PROVIDERS[provider]["api_key"],
                model or PROVIDERS[provider]["default_model"],
                voice or PROVIDERS[provider]["default_voice"],
                emotion,
            )
        return None, None, None, None

    async def close(self):
        await self.async_http_client.close()