from httpx import AsyncClient
import asyncio



async def main():
    url = 'http://127.0.0.1:8080/chat'
    print("Welcome to the chatroom! Start chatting (type 'quit' to exit):")

    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]  # init messages

    while True:
        input_text = input("Please enter your message (type 'quit' to exit):")

        # If the user enters "quit", the program exits.
        if input_text.lower() == 'quit':
            print("Exiting chat...")
            break

        # Add the user's input message to the message list, with the role 'user'
        messages.append({'role': 'user', 'content': input_text})

        async with AsyncClient(timeout=5) as client:
            response = await client.post(url=url, json=messages)
            # If the server response status code is 200, it means the request is successful.
            if response.status_code == 200:
                print(response.json()['content'])
                print()

                # Keep a complete record of the conversation
                messages.append(response.json())
            else:
                print(f"Request failed, status code: {response.status_code}, 'message': {response.text}")


if __name__ == '__main__':
    asyncio.run(main())
