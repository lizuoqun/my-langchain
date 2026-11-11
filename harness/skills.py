import json, os, re, yaml
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

SYSTEM_PROMPT = f"""你是一位甜品店店员，用中文简要回答。

遇到你不了解的内容（例如本店菜单、价格、推荐），必须先调用 load_skill 工具加载对应技能，
拿到技能内容后再回答，不要自己编造。

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
    }
]


def execute_tool(name: str, args: str) -> str:
    args = json.loads(args)
    if name == "load_skill":
        return SKILL_LOADER.get_content(args["skill_name"])
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
            print(f"[调用技能] {name} 参数：{args}")
            tool_output = execute_tool(name, args)
            print(f"[技能加载] 已注入 {len(tool_output)} 个字符的技能正文")
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
