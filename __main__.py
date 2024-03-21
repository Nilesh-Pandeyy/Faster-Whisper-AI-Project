import asyncio
import sys
import threading
from datetime import datetime
from faster_whisper import WhisperModel
from audio_transcriber import AppOptions
from audio_transcriber import AudioTranscriber
from utils.audio_utils import base64_to_audio
from utils.file_utils import read_json, write_json, write_audio
from websoket_server import WebSocketServer
from openai_api import OpenAIAPI

transcriber: AudioTranscriber = None
event_loop: asyncio.AbstractEventLoop = None
thread: threading.Thread = None
websocket_server: WebSocketServer = None
openai_api: OpenAIAPI = None

def get_user_settings():
    current_time=datetime.now()
    timestamp=current_time.strftime("%H:%M:%S.%f")
    print("get_user_settings starting time",timestamp)
    data_types = ["app_settings", "model_settings", "transcribe_settings"]
    user_settings = {}

    try:
        data = read_json("settings", "user_settings")
        for data_type in data_types:
            user_settings[data_type] = data[data_type]
    except Exception as e:
        print(str(e))

    return user_settings

def stop_transcription():
    current_time=datetime.now()
    timestamp=current_time.strftime("%H:%M:%S.%f")
    print("Stop transcription start time ",timestamp)
    global transcriber, event_loop, websocket_server, openai_api
    if transcriber is None:
        return

    # Ensure we're running the stop operations in the event loop of the main thread
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If the loop is running, schedule the coroutines to stop transcription and server
        loop.run_until_complete(transcriber.stop_transcription())
        if websocket_server is not None:
            loop.run_until_complete(websocket_server.stop_server())
    else:
        # If the loop is not running, we directly run the coroutines (less likely scenario)
        asyncio.run(transcriber.stop_transcription())
        if websocket_server is not None:
            asyncio.run(websocket_server.stop_server())


    # Properly close the event loop if it's not already closed
    if not loop.is_closed():
        loop.close()

    # Reset global variables
    transcriber = None
    event_loop = None
    websocket_server = None
    openai_api = None

def start_transcription():
    current_time=datetime.now()
    timestamp=current_time.strftime("%H:%M:%S.%f")
    print("First time called start_transcription", timestamp)
    global transcriber, event_loop, websocket_server, openai_api
    user_settings = get_user_settings()
    #print(user_settings)
    try:
        (
            filtered_app_settings,
            filtered_model_settings,
            filtered_transcribe_settings,
        ) = extracting_each_setting(user_settings)

        whisper_model = WhisperModel(**filtered_model_settings)
        app_settings = AppOptions(**filtered_app_settings)
        event_loop = asyncio.get_event_loop()  # Get the main thread's event loop

        if app_settings.use_websocket_server:
            websocket_server = WebSocketServer(event_loop)
            event_loop.run_until_complete(websocket_server.start_server())

        if app_settings.use_openai_api:
            openai_api = OpenAIAPI()

        transcriber = AudioTranscriber(
            event_loop,
            whisper_model,
            filtered_transcribe_settings,
            app_settings,
            websocket_server,
            openai_api,
        )

        event_loop.run_until_complete(transcriber.start_transcription())
    except Exception as e:
        stop_transcription()
        print(str(e))

def get_filtered_app_settings(settings):
    current_time=datetime.now()
    timestamp=current_time.strftime("%H:%M:%S.%f")
    print("get_filtered_app_settings start time",timestamp)
    valid_keys = AppOptions.__annotations__.keys()
    return {k: v for k, v in settings.items() if k in valid_keys}


def get_filtered_model_settings(settings):
    current_time=datetime.now()
    timestamp=current_time.strftime("%H:%M:%S.%f")
    print("get_filtered_model_settings(settings) start time",timestamp)
    valid_keys = WhisperModel.__init__.__annotations__.keys()
    return {k: v for k, v in settings.items() if k in valid_keys}


def get_filtered_transcribe_settings(settings):
    current_time=datetime.now()
    timestamp=current_time.strftime("%H:%M:%S.%f")
    print("get_filtered_transcribe_settings start time",timestamp)
    valid_keys = WhisperModel.transcribe.__annotations__.keys()
    return {k: v for k, v in settings.items() if k in valid_keys}


def extracting_each_setting(user_settings):
    current_time=datetime.now()
    timestamp=current_time.strftime("%H:%M:%S.%f")
    print("extracting_each_setting start time ",timestamp)
    filtered_app_settings = get_filtered_app_settings(user_settings["app_settings"])
    filtered_model_settings = get_filtered_model_settings(
        user_settings["model_settings"]
    )
    filtered_transcribe_settings = get_filtered_transcribe_settings(
        user_settings["transcribe_settings"]
    )

    write_json(
        "settings",
        "user_settings",
        {
            "app_settings": filtered_app_settings,
            "model_settings": filtered_model_settings,
            "transcribe_settings": filtered_transcribe_settings,
        },
    )

    return filtered_app_settings, filtered_model_settings, filtered_transcribe_settings

def on_close(page):
    print(page, "was closed")

    if transcriber and transcriber.transcribing:
        stop_transcription()
    sys.exit()

if __name__ == "__main__":
    start_transcription()

