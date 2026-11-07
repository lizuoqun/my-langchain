from openai import OpenAI
from openai.types.chat import ChatCompletionMessageFunctionToolCall
import os, json, subprocess
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
            "name": "bash",
            "description": "执行shell命令",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "具体的命令代码（Windows 环境请使用 dir / type / node 等 Windows 原生命令，不要用 ls / cat / pwd 等 bash 命令）"
                    }
                },
                "required": ["command"]
            }
        }
    }
]


def bash(command: str) -> str:
    """执行命令并返回 stdout + stderr（失败时也能看到错误信息）。"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout
    if result.stderr:
        output += ("\n" if output else "") + f"[stderr] {result.stderr}"
    if result.returncode != 0:
        output += f"\n[exit code: {result.returncode}]"
    return output


def execute_tool(name: str, args: str) -> str | None:
    args = json.loads(args)
    if name == "bash":
        return bash(args['command'])
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
            if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                continue
            name, args = tool_call.function.name, tool_call.function.arguments
            tool_output = execute_tool(name, args)
            messages.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': tool_output or '',
            })

    print('完整的messages：')
    for message in messages:
        print(message)
    return messages[-1]['content']


if __name__ == '__main__':
    while True:
        user_input = input("\n用户输入：")
        if user_input == 'exit':
            break
        assistant_output = agent(user_input)
        print("\n模型回答：", assistant_output)
