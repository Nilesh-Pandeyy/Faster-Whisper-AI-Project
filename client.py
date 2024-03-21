import asyncio
import websockets
import sounddevice as sd
import numpy as np
import base64
import sys
from datetime import datetime
# Configuration for audio capture
RATE = 16000  # Sample rate
CHUNK = 1024  # Number of audio samples per frame
CHANNELS = 1  # Mono audio
DTYPE = 'float32'  # Type of data for audio samples
WEBSOCKET_URL = 'ws://localhost:8765'

async def audio_stream(websocket_url):
    # Create an event loop for asynchronous operations
    loop = asyncio.get_event_loop()

    # Regular function to be called with each chunk of audio data
    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        # Convert the NumPy array with audio data to bytes, then to a base64 encoded string
        audio_bytes = indata.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        # Schedule the sending of the encoded audio data over the WebSocket
        loop.call_soon_threadsafe(asyncio.ensure_future, websocket.send(audio_base64))

    # Open a WebSocket connection
    async with websockets.connect(websocket_url) as websocket:
        # Start the audio stream
        stream = sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype=DTYPE, blocksize=CHUNK, callback=callback)
        with stream:
            print("Streaming audio to server. Press Ctrl+C to stop...")
            # Keep the stream open until the user stops the program
            while stream.active:
                try:
                    # Wait for transcription results from the server
                    transcription = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    current_time = datetime.now()
                    print(transcription)
                except asyncio.TimeoutError:
                    # No data received from the server, continue streaming
                    pass
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed")
                    break

        # Optionally, send a "STOP" signal to the server to indicate the end of the stream
        await websocket.send("STOP")

def main():
    websocket_url = WEBSOCKET_URL
    # Run the audio stream coroutine
    asyncio.run(audio_stream(websocket_url))

if __name__ == "__main__":
    main()