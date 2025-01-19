# -*- coding: utf-8 -*-
# @Time    : 18/1/2025 18:32
# @Author  : Zerone
# @File    : httpTest_nonConcurrent.py
# @IDE     ：PyCharm 
# @Description：

import json
import time
from typing import Dict, Any, Tuple, List
from utils_loadingDataset import sample_requests, get_tokenizer

from httpx import AsyncClient


# The test is a local http server
async def send_request(http_client: AsyncClient,
                       url: str,
                       messages: List[Dict[str, Any]],
                       prompt_len: int) -> Dict[str, Any]:
    """
    Sends an HTTP POST request to the specified URL with the given messages.

    Parameters:
    - http_client: An instance of AsyncClient to send the request.
    - url: The target URL for the request.
    - messages: A list of dictionaries containing the messages to be sent.
    - prompt_len: The length of the prompt message.

    Returns:
    - A dictionary containing the response from the server or an error message.
    """
    request_start_time = time.time()
    response = await http_client.post(url=url, json=messages)
    if response.status_code == 200:
        response = response.json()
        output_tokens = response['content']
        request_end_time = time.time()

        REQUEST_LATENCY.append((prompt_len, len(output_tokens), request_end_time - request_start_time))
        return response
    else:
        return {'error': response.status_code, 'message': response.text}


async def benchmark(
        url: str,
        input_requests: List[Tuple[str, int, int]],
) -> None:
    """
    Benchmarks the HTTP POST requests to the specified URL with the given input requests.

    Parameters:
    - input_requests: A list of tuples, where each tuple contains:
        - prompt (str): The input prompt message.
        - prompt_len (int): The length of the prompt message.
        - output_len (int): The expected length of the output message.

    Returns:
    - None
    """
    async with AsyncClient(timeout=60) as http_client:
        for idx, request in enumerate(input_requests):
            messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]  # init messages
            prompt, prompt_len, output_len = request
            messages.append({'role': 'user', 'content': prompt})

            response = await send_request(http_client=http_client, url=url, messages=messages, prompt_len=prompt_len)
            print(f"Response {idx + 1}: {json.dumps(response, ensure_ascii=False, indent=2)}")


if __name__ == '__main__':
    import logging
    import asyncio
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser(description="HTTP POST stress test with non-concurrent.")
    parser.add_argument('--url', type=str, default='http://127.0.0.1:8080/chat', help='The url of the server')
    parser.add_argument('--dataset_path', type=str, default='./ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json',
                        help='Dataset path of stress test')
    parser.add_argument('--tokenizer_path', type=str, default='./Qwen1.5-0.5B-Chat',
                        help='Tokenizer path used for tokenizing the input requests')
    parser.add_argument('--num_request', type=int, default=10, help='The number of requests to be sent')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    REQUEST_LATENCY: List[Tuple[int, int, float]] = []  # Tuple[prompt_len, output_len, latency], global variable

    # Prepare for the testing dataset
    logging.info("Preparing for Sequential non-concurrent http post.")
    tokenizer = get_tokenizer(args.tokenizer_path)
    input_requests = sample_requests(args.dataset_path, args.num_request, tokenizer)
    print(f'A example of input request: {input_requests[0]}')

    # Start the testing
    logging.info("Sequential non-concurrent http post starts.")
    start_time = time.time()
    asyncio.run(benchmark(url=args.url, input_requests=input_requests))
    end_time = time.time()

    # print the final test result
    total_test_time = end_time - start_time
    print(f"Total test time: {total_test_time:.2f} s")

    print(
        f"Throughput: {len(REQUEST_LATENCY) / total_test_time:.2f} requests/s")  # Calculate the request throughput per second

    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")  # Average request latency

    # Calculate the average latency per token (including prompt tokens and generated tokens)
    # avg_per_token_latency = np.mean(
    #     [
    #         latency / (prompt_len + output_len)
    #         for prompt_len, output_len, latency in REQUEST_LATENCY
    #     ]
    # )
    # print(f"Average latency per token: {avg_per_token_latency:.2f} s")

    # Calculate the average latency per generated token (only generated output tokens are counted)
    # avg_per_output_token_latency = np.mean(
    #     [latency / output_len for _, output_len, latency in REQUEST_LATENCY]
    # )
    # print("Average latency per output token: " f"{avg_per_output_token_latency:.2f} s")
