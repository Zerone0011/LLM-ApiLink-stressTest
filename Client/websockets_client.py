# -*- coding: utf-8 -*-
# @Time    : 18/1/2025 18:00
# @Author  : Zerone
# @File    : websockets_client.py
# @IDE     ：PyCharm
# @Description：

import asyncio
import random

import websockets


async def chat(natural_typing_speed: bool = True):
    # ws:// indicates the use of the WebSocket protocol,
    # 127.0.0.1 indicates the local address,
    # 8080 is the port number, and /chat is the path
    url = "ws://127.0.0.1:8080/chat"

    async with websockets.connect(url) as websocket:
        print("Welcome to the chatroom! Start chatting (type 'quit' to exit):")

        try:
            while True:
                message = input("Please enter your message (type 'quit' to exit):")

                # If the user enters "quit", the program exits.
                if message.lower() == "quit":
                    await websocket.send("quit")
                    print("Exiting chat...")
                    break

                # Send the user input message to the server
                await websocket.send(message)

                while True:
                    response = await websocket.recv()

                    # The server uses '&&&endofsection&&&' to mark the end of the complete response
                    if response.endswith('&&&endofsection&&&'):
                        print()
                        break
                    else:
                        # end='' means no line break,
                        # flush=True ensures that the output is flushed to the console immediately
                        # print(response, end='', flush=True)
                        for char in response:
                            print(char, end='', flush=True)
                            if natural_typing_speed:
                                await asyncio.sleep(random.uniform(0.03, 0.1))  # Simulate natural typing speed
        except websockets.exceptions.ConnectionClosed:
            print("The connection is closed.")

if __name__ == "__main__":
    asyncio.run(chat(natural_typing_speed=False))
