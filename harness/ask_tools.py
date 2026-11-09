import re, os, json, subprocess
from fnmatch import fnmatch
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
    },
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

rules_dict = {
    "grep": "allow",
    "read_file": "allow",
    "write_file": "ask",
    "edit_file": "ask",
    "bash": "ask"
}

allow_set = set()

# 危险操作指令，这些不允许执行
danger_pattern = re.compile(
    r"\b(rm|sudo|su|chmod|chown|mkfs|format|dd|shutdown|reboot|halt|poweroff|del|rd|rmdir|truncate|unlink)\b",
    re.IGNORECASE
)


def check_permission(name, args) -> str:
    """
    检查权限，按照：静态规则 -> 会话allow set -> 高危操作硬拒绝 -> 默认ask 【顺序检查】
    """

    args = json.loads(args)

    if rules_dict.get(name) == "allow":
        return "allow"

    for tool_name, pattern in allow_set:
        if tool_name == name and fnmatch(args.get('command', args.get('path', "")), pattern):
            return "allow"

    if name == 'bash' and danger_pattern.search(args.get(args.get('command'), '')):
        return "deny"

    return "ask"


def pattern_of(tool_calls) -> str:
    args = json.loads(tool_calls.arguments)
    target = args.get("command", args.get("path", "")).strip()

    if not target:
        return "*"

    parts = target.split()

    if not parts:
        return "*"

    if tool_calls.name == "bash":
        return f""

    if tool_calls.name in ['write_file', "edit_file"]:
        path = args.get("path")
        if path and not any(char in path for char in '*?[]'):
            dirname = os.path.dirname(path)
            if dirname:
                return f"{dirname}/*"
            return path if '/' not in path else '/*'

    return ''


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

    if name == "grep":
        return grep(args['path'], args['pattern'])
    elif name == "read_file":
        return read_file(args['path'], args['offset'], args['limit'])
    elif name == "write_file":
        return write_file(args['path'], args['content'])
    elif name == "edit_file":
        return edit_file(args['path'], args['old_string'], args['new_string'])
    elif name == "bash":
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
            permission = check_permission(name, args)
            if permission == 'deny':
                tool_output = '权限不足，请勿执行危险操作。'
            elif permission == 'ask':
                user_choose = input(f"工具：{name}调用，参数为{args}， 是否执行？Y/N\n").strip().upper()
                if user_choose == 'Y':
                    allow_set.add((name, pattern_of(tool_call.function)))
                    tool_output = execute_tool(name, args)
                else:
                    tool_output = '用户取消执行。'
            else:
                tool_output = execute_tool(name, args)

            print(f"执行：{name}，{args}----allow_set：{allow_set}")
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

# 在test目录下，帮我修改test.js文件当中的bug
