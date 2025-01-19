# LLM API Link Local Deployment and Stress Test
This project is a simple demonstration of how to **deploy the LLM API Link locally** and how to stress test local-server-url under **non-concurrent** and **concurrent** product environment.
## Required environment
- Follow the instructions on the [FastAPI website](https://fastapi.tiangolo.com/zh/tutorial/) to install fastapi and uvicorn.
- Make sure you have **python3** installed on your machine and python library listed in requirments.txt.
    ```shell
    # you can easily install required python library by running the following command
    pip install -r requirements.txt
    ```

## Pre-requisite
### 1. Remote LLM API Link
- You need to have a remote LLM **API key**. I uesd Alibaba Cloud (bailian) LLM API Link for this demonstration. 
You can register or enroll from this [link](https://www.aliyun.com/product/bailian) for a personal account to get the API key.
- Then you need to set the API key in environment variable.
  1. create a file named **.env** in the root directory of the project.
  2. Add the following line in the **.env** file.
  ```shell
    KPI_KEY_bailian=your_api_key
    ```
### 2. Dataset for stress test
- I used a sample dataset for this demonstration. You can use any dataset for this demonstration.
- The dataset I used can be downloaded from [Hugging face](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main).
Download the dataset named ```ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json``` and place it in the **Dataset** folder of the project.
### 3. Tokenizer
  - I used the **Qwen1.5-0.5B-Chat** tokenizer for this demonstration. You can use any tokenizer for you like.
    - The tokenizer I used can be downloaded from [Hugging face](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat).
  ```shell
  # you can easily download the tokenizer by running the following command
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat`
  ```
## How to run it
### 1. Local Server
- Run the following command to start the local server.
```shell
python server.py
```
### 2. Http connection and websocket connection
- Run the following command to start the **Http** connection or **Websocket** connection.
```shell
# for Http connection
python Client/http_client.py
```
```shell
# for Websocket connection
python Client/websockets_client.py
```
### 3. Stress Test
- Run the following command to start the Http stress test in **non-concurrent** or **concurrent** simulated environment.
```shell
# for non-concurrent stress test
python HttpTest/httpTest_nonConcurrent.py
```
```shell
# for concurrent stress test
python HttpTest/httpTest_Concurrent.py
```
## Reference Projects
Thanks to the following projects for providing the inspiration for this project.
- [Bilibili-使用fastapi搭建ChatGPT对话后台](https://www.bilibili.com/video/BV18L41117Dn/?vd_source=e08457a124beb689f3883e51b8aa43a5)
- [LLM压测](https://blog.csdn.net/liuzhenghua66/article/details/139332747)
