import asyncio
import functools
import queue
import numpy as np
from datetime import datetime
from typing import NamedTuple
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
import json
from vad import Vad
from utils.file_utils import write_audio
from websoket_server import WebSocketServer
from openai_api import OpenAIAPI

import base64

class AppOptions(NamedTuple):
    audio_device: int
    silence_limit: int = 8
    noise_threshold: int = 5
    non_speech_threshold: float = 0.1
    include_non_speech: bool = False
    create_audio_file: bool = True
    use_websocket_server: bool = False
    use_openai_api: bool = False


class AudioTranscriber:
    def __init__(
        self,
        event_loop: asyncio.AbstractEventLoop,
        whisper_model: WhisperModel,
        transcribe_settings: dict,
        app_options: AppOptions,
        websocket_server: WebSocketServer,
        openai_api: OpenAIAPI,
    ):
        self.event_loop = event_loop
        self.whisper_model: WhisperModel = whisper_model
        self.transcribe_settings = transcribe_settings
        self.app_options = app_options
        self.websocket_server = websocket_server
        self.websocket_server.message_handler = self.handle_websocket_message
        self.openai_api = openai_api
        self.vad = Vad(app_options.non_speech_threshold)
        self.silence_counter: int = 0
        self.audio_data_list = []
        self.all_audio_data_list = []
        self.audio_queue = queue.Queue()
        self.transcribing = False
        self.stream = None
        self._running = asyncio.Event()
        self._transcribe_task = None

    async def transcribe_audio(self):
        # Ignore parameters that affect performance
        transcribe_settings = self.transcribe_settings.copy()
        transcribe_settings["without_timestamps"] = True
        transcribe_settings["word_timestamps"] = False
        
        with ThreadPoolExecutor() as executor:
            while self.transcribing:
                try:
                    start_time = datetime.now()
                    audio_data = await self.event_loop.run_in_executor(
                        executor, functools.partial(self.audio_queue.get, timeout=3.0)
                    )

                    # Create a partial function for the model's transcribe method
                    func = functools.partial(
                        self.whisper_model.transcribe,
                        audio=audio_data,
                        **transcribe_settings,
                    )

                    # Run the transcribe method in a thread
                    segments, _ = await self.event_loop.run_in_executor(executor, func)
                    
                    for segment in segments:
                        # Create a dictionary with the transcription text and the start time
                        transcription_entry = {
                            "time": start_time.strftime("%H:%M:%S.%f"),
                            "translatedText": segment.text
                        }
                        # Convert the dictionary to a JSON string
                        json_output = json.dumps(transcription_entry, ensure_ascii=False)

                        # Print the JSON string to the CLI
                        print(json_output)

                    # If a WebSocket server is being used, send the JSON string to the client
                    if self.websocket_server is not None:
                        await self.websocket_server.send_message(json_output)

                    end_time = datetime.now()
                    print("Transcription finished at:", end_time.strftime("%H:%M:%S.%f"))
                    print("Total transcription time:", (end_time - start_time).total_seconds(), "seconds")

                except queue.Empty:
                    # Skip to the next iteration if a timeout occurs
                    continue
                except Exception as e:
                    print(str(e))


    def process_audio(self, audio_data: np.ndarray, frames: int, time, status):
        is_speech = self.vad.is_speech(audio_data)
        if is_speech:
            self.silence_counter = 0
            self.audio_data_list.append(audio_data.flatten())
        else:
            self.silence_counter += 1
            if self.app_options.include_non_speech:
                self.audio_data_list.append(audio_data.flatten())

        if not is_speech and self.silence_counter > self.app_options.silence_limit:
            self.silence_counter = 0

            if self.app_options.create_audio_file:
                self.all_audio_data_list.extend(self.audio_data_list)

            if len(self.audio_data_list) > self.app_options.noise_threshold:
                concatenate_audio_data = np.concatenate(self.audio_data_list)
                self.audio_data_list.clear()
                self.audio_queue.put(concatenate_audio_data)
            else:
                # noise clear
                self.audio_data_list.clear()

    async def handle_websocket_message(self, message):
        # Decode the base64 audio data
        audio_data_bytes = base64.b64decode(message)
        # Convert the bytes data to a NumPy array
        audio_np = np.frombuffer(audio_data_bytes, dtype=np.float32)
        self.process_audio(audio_np, frames=None, time=None, status=None)

    async def start_transcription(self):
        try:
            start_time = datetime.now()
            print("Start transcription method called at:", start_time.strftime("%H:%M:%S.%f"))
            self.transcribing = True
            self._running.set()
            self._transcribe_task = asyncio.run_coroutine_threadsafe(
                self.transcribe_audio(), self.event_loop
            )
            
            while self._running.is_set():
                await asyncio.sleep(1)
            end_time = datetime.now()
            print("Total time taken to start transcription:", end_time - start_time)

        except Exception as e:
            print(str(e))
    
    async def stop_transcription(self):
        try:
            self.transcribing = False
            if self._transcribe_task is not None:
                self.event_loop.call_soon_threadsafe(self._transcribe_task.cancel)
                self._transcribe_task = None

            if self.app_options.create_audio_file and len(self.all_audio_data_list) > 0:
                audio_data = np.concatenate(self.all_audio_data_list)
                self.all_audio_data_list.clear()
                write_audio("web", "voice", audio_data)
                self.batch_transcribe_audio(audio_data)

            # Stop the WebSocket server
            await self.websocket_server.stop_server()
            self._running.clear()
            print("Transcription stopped.")
        except Exception as e:
            print(str(e))
