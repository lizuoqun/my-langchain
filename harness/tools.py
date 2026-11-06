from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = OpenAI(
    base_url=DEFAULT_API_BASE,
    api_key=DEEPSEEK_API_KEY
)

tool_list: list = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市（地点）的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["location"]
            }
        }
    }
]


def get_weather(location: str) -> str:
    return f"{location}的天气是晴天"


def execute_tool(name: str, args: str) -> str | None:
    args = json.loads(args)
    if name == "get_weather":
        return get_weather(args['location'])
    return None


messages: list = [{'role': 'system', 'content': '简要回答'}]


def agent(query: str) -> str:
    messages.append({'role': 'user', 'content': query})
    while True:
        response = model.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            tools=tool_list,
        )
        reply = response.choices[0].message
        messages.append({'role': reply.role, 'content': reply.content, 'tool_calls': reply.tool_calls})

        if not reply.tool_calls:
            break
        for tool_call in reply.tool_calls:
            # print("Tool Call:", tool_call)
            # Tool Call: ChatCompletionMessageFunctionToolCall(id='call_00_ET_vXPEBoOkTXucJKlrFqC61470', function=Function(arguments='{"location": "北京"}', name='get_weather'), type='function', index=0)
            name, args = tool_call.function.name, tool_call.function.arguments
            # print(f"Tool Call: {name}, {args}")
            tool_output = execute_tool(name, args)
            # print("Tool Output:", tool_output)
            messages.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': tool_output or '',
            })

    print('完整的messages：', messages)
    return messages[-1]['content']


if __name__ == '__main__':
    while True:
        user_input = input("用户输入：")
        if user_input == 'exit':
            break
        assistant_output = agent(user_input)
        print("模型回答：", assistant_output)
