# -*- coding: utf-8 -*-
# @Time    : 18/1/2025 21:18
# @Author  : Zerone
# @File    : httpTest_Concurrent.py
# @IDE     ：PyCharm 
# @Description：
import asyncio
import json
import time
from typing import List, Dict, Any, Tuple

from aiohttp import ClientSession

from utils_loadingDataset import sample_requests, get_tokenizer
import aiohttp
from httpx import AsyncClient


async def send_request(http_client: ClientSession,
                       url: str,
                       messages: List[Dict[str, Any]],
                       prompt_len: int) -> Dict[str, Any]:
    """
    Sends an HTTP POST request to the specified URL with the given messages.

    Parameters:
    - http_client: An instance of ClientSession to send the request.
    - url: The target URL for the request.
    - messages: A list of dictionaries containing the messages to be sent.
    - prompt_len: The length of the prompt message.

    Returns:
    - A dictionary containing the response from the server or an error message.
    """
    request_start_time = time.time()
    async with http_client.post(url=url, json=messages) as response:
        if response.status == 200:
            response = await response.json()
            output_tokens = response['content']
            request_end_time = time.time()

            REQUEST_LATENCY.append((prompt_len, len(output_tokens), request_end_time - request_start_time))
            return response
        else:
            return {'error': response.status, 'message': await response.text()}

class BenchMarkRunner:
    """
    BenchMarkRunner is a class designed to manage and execute concurrent HTTP POST requests.

    Attributes:
    - requests: A list of tuples, where each tuple contains:
        - prompt (str): The input prompt message.
        - prompt_len (int): The length of the prompt message.
        - output_len (int): The expected length of the output message.
    - concurrency: The number of concurrent requests to be handled.

    Methods:
    - run: Manages the creation and execution of worker tasks to process the requests.
    - worker: Processes individual requests from the request queue and sends them using an HTTP client.
    """
    def __init__(
        self,
        requests: List[Tuple[str, int, int]],  # prompt, prompt_len, output_len
        concurrency: int,
    ):
        self.concurrency = concurrency
        self.requests = requests
        self.request_left = len(requests)   # Initialize the number of remaining requests

        # Create an asynchronous queue to manage the distribution of requests
        self.request_queue = asyncio.Queue(concurrency or 100)

    async def run(self):
        tasks = []  # Used to store all work tasks

        for i in range(self.concurrency):
            # Use asyncio.create_task() to create and execute immediately an asynchronous task
            tasks.append(asyncio.create_task(self.worker()))

        # must put all requents into queue behind the worker start
        # If the worker has not been started, requests placed in the queue will pile up.
        # If the queue has a capacity limit, the queue may fill up before the worker starts, causing the producer (request enqueue) to block.
        # Add all requests to the request queue
        for req in self.requests:
            await self.request_queue.put(req)

        # Add a termination signal to each worker
        for _ in range(self.concurrency):
            await self.request_queue.put(('&&&No_more_tasks&&&', 0, 0))  # Termination signal

        # Wait for all tasks to complete and return when all tasks are completed
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        # await asyncio.gather(*tasks)  # Same function as the previous one

    async def worker(self):
        timeout = aiohttp.ClientTimeout(total=5 * 60)
        # (aiohttp.ClientSession) allow you to switch to other tasks
        # while waiting for a response without blocking the entire program.
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:

                messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]  # init messages
                prompt, prompt_len, completion_len = await self.request_queue.get() # Get a request from the request queue
                messages.append({'role': 'user', 'content': prompt})

                if prompt == '&&&No_more_tasks&&&':  # If a termination signal is received, the task ends
                    break

                response = await send_request(http_client=session, url=url, messages=messages, prompt_len=prompt_len)

                self.request_left -= 1
                print(f"Response {len(self.requests) - self.request_left}: {json.dumps(response, ensure_ascii=False, indent=2)}")

if __name__ == '__main__':
    import logging
    import asyncio
    import numpy as np

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    REQUEST_LATENCY: List[Tuple[int, int, float]] = []  # Tuple[prompt_len, output_len, latency]

    url = "http://127.0.0.1:8080/chat"
    dataset_path = r'./ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
    logging.info("Preparing for concurrent http post.")

    tokenizer_name_or_path = './Qwen1.5-0.5B-Chat'
    num_request = 10
    concurrency = 5

    tokenizer = get_tokenizer(tokenizer_name_or_path)

    # Sample some requests from the dataset
    input_requests = sample_requests(dataset_path, num_request, tokenizer)
    print(f'A example of input request: {input_requests[0]}')

    logging.info("Concurrent http post starts.")

    start_time = time.time()
    asyncio.run(BenchMarkRunner(input_requests, concurrency).run())
    end_time = time.time()

    total_test_time = end_time - start_time
    print(f"Total test time: {total_test_time:.2f} s")

    # Calculate the request throughput per second
    print(f"Throughput: {len(REQUEST_LATENCY) / total_test_time:.2f} requests/s")

    # Average request latency
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")

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