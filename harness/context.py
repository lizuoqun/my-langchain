import json
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
System_Message = {'role': 'system', 'content': '稍微简要一些回答'}

TOKEN_LIMIT = 100
COMPACT_AT = 0.80
KEEP_RECENT = 2


def estimate_tokens(text: str) -> int:
    return len(text) // 3


def total_token(messages: list):
    return sum([estimate_tokens(m['content']) for m in messages])


def compact(messages: list) -> list:
    head, tail = messages[:-KEEP_RECENT], messages[-KEEP_RECENT:]
    prompt = '把以下对话内容进行摘要'
    response_prompt = model.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[System_Message, {'role': 'user', 'content': prompt + serialize(head)}]  # type: ignore[list-item]
    )
    reply = response_prompt.choices[0].message
    summary = reply.content
    return [System_Message, {'role': 'assistant', 'content': summary}, *tail]


def serialize(serialize_messages: list):
    out = []
    for msg in serialize_messages:
        m = {'role': msg['role'], 'content': msg.get('content', '')}
        tool_calls = msg.get('tool_calls', None)
        if tool_calls:
            m['tool_calls'] = [{'name': tool_call.function.name, 'arguments': tool_call.function.arguments} for
                               tool_call in tool_calls]
        out.append(m)
    return json.dumps(out)


messages: list = []


def agent(query: str):
    messages.append({'role': 'user', 'content': query})
    while True:
        if total_token(messages) > TOKEN_LIMIT * COMPACT_AT:
            messages[:] = compact(messages)
        response = model.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages
        )
        reply = response.choices[0].message
        messages.append({'role': reply.role, 'content': reply.content})

        if not reply.tool_calls:
            break
    print('完整的messages：')
    for msg in messages:
        print(msg)
    return messages[-1]['content']


if __name__ == '__main__':
    while True:
        user_input = input("用户输入：")
        if user_input == 'exit':
            break
        assistant_output = agent(user_input)
        print("模型回答：", assistant_output)
