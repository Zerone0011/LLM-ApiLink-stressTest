import json
from typing import Union, Dict, AsyncIterable, Any
from httpx import AsyncClient


async def stream_request(url: str, headers: Dict[str, str], params: Union[str, Dict[str, Any]]) -> AsyncIterable[Dict[str, Any]]:
    """
    Parameters:
    - url: The target URL for the request.
    - headers: Request headers in dictionary format.
    - params: Request parameters, which can be a string or dictionary.

    Function:
    - Sends a POST request with streaming response to the specified URL.
    - Parses the server's response line by line and returns the parsed results asynchronously.

    yield example:
    {'content': '', 'role': 'assistant'}
    {'content': '我是'}
    """
    async with AsyncClient() as client:
        async with client.stream("POST", url, headers=headers, json=params, timeout=60) as lines:
            async for line in lines.aiter_lines():

                if not line.strip():  # If the current line is blank, skip it.
                    continue
                line = line.replace("data: ", "")  # Remove the "data: " prefix at the beginning of the line
                try:
                    response = json.loads(line)
                except Exception:
                    # json parsing failed
                    response = {"choices": [{"finish_reason": "json convertion error"}]}

                # Check if the build is complete
                if response.get("choices")[0].get("finish_reason") is not None:
                    yield response.get("choices")[0].get("delta")  # return final response
                    return

                # Extract the API response and return
                yield response.get("choices")[0].get("delta")


async def http_request(url: str, headers: Dict[str, str], params: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parameters:
        - url: the target URL of the request.
        - headers: request header information, in dictionary format.
        - params: request parameters, which can be string or dictionary format.

    Function:
        - Send a POST request to the specified URL.
        - Return the parsed JSON response data.

    Return value:
        - The parsed JSON response data, in dictionary format.

    Return example:
        {'role': 'assistant', 'content': '我是来自阿里云的大规模语言模型'}
    """
    async with AsyncClient(timeout=30) as client:
        response = await client.post(url=url, headers=headers, json=params)
        return response.json().get("choices")[0].get("message")


if __name__ == '__main__':
    from dotenv import load_dotenv
    import os
    import asyncio

    load_dotenv()
    KPI_key = os.getenv("KPI_KEY_bailian")
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    # 定义请求头和请求体数据
    headers = {
        "Authorization": "Bearer " + KPI_key,
        "Content-Type": "application/json"
    }
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '你是谁？'}]
    params = {
        "stream": False,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 1000,
        "model": "qwen-1.8b-chat",
        "n": 1  # Returns a generated result
    }
    # test stream=False
    response = asyncio.run(http_request(url=url, headers=headers, params=params))
    print(response)

    # test stream=True
    # async def stream_request_test(url: str, headers: Dict[str, str], params: Union[str, Dict[str, Any]]):
    #     async for line in stream_request(url=url, headers=headers, params=params):
    #         print(line)
    #         pass
    #
    # asyncio.run(stream_request_test(url=url, headers=headers, params=params))