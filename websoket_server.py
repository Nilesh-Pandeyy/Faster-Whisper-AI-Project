import asyncio
import websockets
import webbrowser
import os
from typing import Optional

python_root_dir = os.path.dirname(os.path.abspath(__file__))
app_root_dir = os.path.dirname(python_root_dir)


class WebSocketServer:
    def __init__(self, loop, message_handler=None):
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.loop = loop
        self.server = None
        self.message_handler = message_handler

    async def start_server(self):
        print("starting server")
        self.server = await websockets.serve(self.handler, "localhost", 8765)
        #self.call_websocket_client()

    def call_websocket_client(self):
        path = os.path.join(app_root_dir, "websocket_client", "websocket_client.html")
        webbrowser.open("file://" + path)

    async def handler(self, ws, path):
        self.websocket = ws
        print("WebSocket connection established")
        try:
            async for message in ws:
                if self.message_handler:
                    await self.message_handler(message)  # Use the provided message handler
        except Exception as e:
            print(f"Error in WebSocket handler: {e}")
        finally:
            self.websocket = None
            print("WebSocket connection closed")

    async def stop_server(self):
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()

    async def send_message(self, message: str):
        if self.websocket is not None:
            await self.websocket.send(message)

    def send_message_threadsafe(self, message: str):
        if self.websocket is not None:
            asyncio.run_coroutine_threadsafe(self.send_message(message), self.loop)