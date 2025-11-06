import asyncio
from websockets.asyncio.client import connect

async def main():
    async with connect("ws://localhost:8765") as websocket:
        while True:
            await websocket.send("Hi there!")
            response = await websocket.recv()
            print(response)
            await asyncio.sleep(1)

asyncio.run(main())
