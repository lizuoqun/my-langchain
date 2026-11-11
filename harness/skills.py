import json, os, re, urllib.request, yaml
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageFunctionToolCall

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = OpenAI(
    base_url=DEFAULT_API_BASE,
    api_key=DEEPSEEK_API_KEY
)

# 技能目录：harness/skills，下面的每个子目录放一个 SKILL.md
SKILLS_DIR = Path(__file__).parent / "skills"


class SkillLoader:
    """扫描技能目录，把每个 SKILL.md 的元信息（frontmatter）和正文分别存起来"""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills: dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        if not self.skills_dir.exists():
            return
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict, str]:
        # 兼容 Windows 的 \r\n 换行
        match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """拼成给模型看的技能清单（只有名字和描述，不含正文）"""
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """按名字取出某个技能的完整正文"""
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f'<skill name="{name}">\n{skill["body"]}\n</skill>'


SKILL_LOADER = SkillLoader(SKILLS_DIR)

SYSTEM_PROMPT = f"""你是一个乐于助人的助手，用中文简要回答。

遇到你不了解的领域，必须先调用 load_skill 工具加载对应技能，拿到技能内容后再回答，不要自己编造。
技能正文里可能会要求你调用其它工具（例如 http_get 发请求），请严格按要求执行。

当前可用技能：
{SKILL_LOADER.get_descriptions()}"""

tool_list: list = [
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "加载指定技能的详细知识内容，回答相关问题前必须先调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "技能名称，必须是系统提示中列出的可用技能之一"
                    }
                },
                "required": ["skill_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "向指定 URL 发送 GET 请求并返回响应内容，用于获取技能中要求的实时数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "完整的请求地址，必须以 http:// 或 https:// 开头"},
                    "max_chars": {"type": "integer", "description": "最大返回字符数，默认 4000"}
                },
                "required": ["url"]
            }
        }
    }
]


def http_get(url: str, max_chars: int = 4000) -> str:
    """真正发请求的地方，技能本身不执行代码，由这里执行"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"Error fetching {url}: {e}"
    return raw[:max_chars]


def execute_tool(name: str, args: str) -> str:
    args = json.loads(args)
    if name == "load_skill":
        return SKILL_LOADER.get_content(args["skill_name"])
    if name == "http_get":
        return http_get(args["url"], args.get("max_chars", 4000))
    return f"未知工具：{name}"


messages: list = [
    {"role": "system", "content": SYSTEM_PROMPT}
]


def agent(query: str) -> str:
    messages.append({"role": "user", "content": query})
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
            print(f"[调用工具] {name} 参数：{args}")
            tool_output = execute_tool(name, args)
            print(f"[工具返回] {tool_output[:120]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_output,
            })

    print('完整的messages：')
    for message in messages:
        print(message)
    return messages[-1]["content"]  # type: ignore[return-value]


if __name__ == '__main__':
    while True:
        user_input = input("用户输入：")
        if user_input == 'exit':
            break
        assistant_output = agent(user_input)
        print("模型回答：", assistant_output)
