# -*- coding: utf-8 -*-
# @Time    : 18/1/2025 18:00
# @Author  : Zerone
# @File    : server.py
# @IDE     ：PyCharm
# @Description：

import logging
from collections import defaultdict
from typing import List
import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, APIRouter, Body
from starlette.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from utils import *

# setting logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

'''
CORSMiddleware 是 FastAPI 提供的一个中间件，用于解决 跨域资源共享（CORS） 问题。

跨域请求是指 客户端和服务器的域名、端口或协议不一致时，客户端向服务器发送的 HTTP 请求。
浏览器默认会阻止 跨域请求，因此需要使用 CORS 机制来允许跨域访问。
'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源的跨域请求
    allow_credentials=True,  # 允许发送跨域请求时携带身份验证信息
    allow_methods=["*"],  # 允许所有的 HTTP 方法（GET、POST、PUT 等）
    allow_headers=["*"],  # 允许所有的 HTTP 头
)

load_dotenv()
KPI_key = os.getenv("KPI_KEY_bailian")
url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
headers = {
        "Authorization": "Bearer " + KPI_key,
        "Content-Type": "application/json"
    }

@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    try:
        await websocket.accept()
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]  # init messages

        while True:
            input_text = await websocket.receive_text()
            # Add the client's message to the chat history
            messages.append({"role": "user", "content": input_text})

            # If the user sends "quit", close the WebSocket connection
            if input_text == "quit":
                await websocket.close()
                logging.info(messages)
                break

            params = {
                "stream": True,
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 1000,
                "model": "qwen-1.8b-chat",
                "n": 1  # Returns a generated result
            }

            result = defaultdict(str) # init result dict

            # Get streaming responses from an API
            async for line in stream_request(url=url, headers=headers, params=params):
                role = line.get("role")
                content = line.get("content")

                if role:
                    result["role"] = role

                if content:
                    await websocket.send_text(content)  # Send generated content in time
                    result["content"] += content  # Concatenate API streaming responses

            messages.append(dict(result))
            await websocket.send_text('&&&endofsection&&&') # Send an end flag to indicate the end of the response
            logging.info(messages)  # Print conversation history
    except WebSocketDisconnect:
        return

'''
HTTP requests are short-lived, so we can’t write a while True loop like we do with websocker. 
Therefore, the function of saving historical data needs to be implemented on the client side.
'''
@app.post("/chat")
async def chat(messages: List[Dict[str, str]] = Body(...)):
    logging.info(f"Received message: {messages}")
    params = {
        "stream": False,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 1000,
        "model": "qwen-1.8b-chat",
        "n": 1  # Returns a generated result
    }

    # Sending user messages to the remove API
    return await http_request(url=url, headers=headers, params=params)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)