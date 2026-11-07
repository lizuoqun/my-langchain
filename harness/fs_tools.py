import re, os, json

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageFunctionToolCall
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
            "name": "grep",
            "description": "按照正则搜索文件，返回匹配的行号以及对应行的内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "正则表达式"
                    }
                },
                "required": ["path", "pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定文件内容（带行号），大文件用offset、limit分段",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "起始行号"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "行数限制"
                    }
                },
                "required": ["path", "offset", "limit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "覆盖方式写入文件，修改已有文件优先使用edit_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "content": {
                        "type": "string",
                        "description": "文件内容"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "精确替换文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "要替换的旧字符串"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "新字符串"
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    }
]


def grep(path: str, pattern: str) -> str:
    output = []
    for row, line_content in enumerate(open(path, encoding='utf-8').read().splitlines(), start=1):
        if re.search(pattern, line_content):
            output.append(f"行号：{row + 1} | {line_content}")
    return '\n'.join(output)


def read_file(path: str, offset: int, limit: int) -> str:
    lines = open(path, encoding='utf-8').read().splitlines()
    return '\n'.join(
        f"{row}|{line_content}" for row, line_content in enumerate(lines[offset - 1:offset - 1 + limit], start=offset))


def write_file(path: str, content: str) -> str:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"文件 {path} 已成功覆盖写入。"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        file_content = f.read()
        file_count = file_content.count(old_string)
    file_content = file_content.replace(old_string, new_string)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(file_content)
    return f"文件 {path} 已成功编辑。字符串{old_string}已替换为{new_string}，一共替换了{file_count}个。"


def execute_tool(name: str, args: str) -> str | None:
    args = json.loads(args)

    if name == "grep":
        return grep(args['path'], args['pattern'])
    elif name == "read_file":
        return read_file(args['path'], args['offset'], args['limit'])
    elif name == "write_file":
        return write_file(args['path'], args['content'])
    elif name == "edit_file":
        return edit_file(args['path'], args['old_string'], args['new_string'])
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


# 在test目录下，用js代码实现一个斐波拉契的方法，写入到fibonacci.js文件当中