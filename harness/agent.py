from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = OpenAI(
    base_url=DEFAULT_API_BASE,
    api_key=DEEPSEEK_API_KEY
)

messages: list = [{'role': 'system', 'content': '简要回答'}]


def agent(query: str):
    messages.append({'role': 'user', 'content': query})
    while True:
        response = model.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages
        )
        reply = response.choices[0].message
        messages.append({'role': reply.role, 'content': reply.content})

        if not reply.tool_calls:
            break
    print('完整的messages：', messages)
    return messages[-1]['content']


if __name__ == '__main__':
    while True:
        user_input = input("用户输入：")
        if user_input == 'exit':
            break
        assistant_output = agent(user_input)
        print("模型回答：", assistant_output)
