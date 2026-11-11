import json, os, re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = OpenAI(
    base_url=DEFAULT_API_BASE,
    api_key=DEEPSEEK_API_KEY
)

MODEL_NAME = "deepseek-v4-flash"

# 记忆目录：harness/memory
MEMORY_DIR = Path(__file__).parent / "memory"

# history 超过这个条数就触发压缩，压缩后只保留最近 RECENT_MESSAGES 条原文
COMPACT_AFTER = 10
RECENT_MESSAGES = 4


class MemoryStore:
    """管理四份记忆文件：
    MEMORY.md        长期记忆（跨会话，压缩时由模型重写）
    USER.md          用户画像（跨会话）
    history.jsonl    全量原始对话流水（只追加，方便审计/回溯）
    YYYY-MM-DD.md    当天的情景记忆（压缩时由模型追加）
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.memory_file = memory_dir / "MEMORY.md"
        self.user_file = memory_dir / "USER.md"
        self.history_file = memory_dir / "history.jsonl"

    def ensure_files(self):
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            self.memory_file.write_text("# 长期记忆\n\n", encoding="utf-8")
        if not self.user_file.exists():
            self.user_file.write_text("# 用户画像\n\n", encoding="utf-8")
        if not self.history_file.exists():
            self.history_file.touch()

    def append_history(self, message: dict):
        """原始对话只追加不修改，作为最底层的持久化记录"""
        self.ensure_files()
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "role": message.get("role"),
            "content": message.get("content"),
        }
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def read_memory(self) -> str:
        self.ensure_files()
        return self.memory_file.read_text(encoding="utf-8")

    def write_memory(self, text: str):
        self.ensure_files()
        self.memory_file.write_text(text.strip() + "\n", encoding="utf-8")

    def read_user(self) -> str:
        self.ensure_files()
        return self.user_file.read_text(encoding="utf-8")

    def write_user(self, text: str):
        self.ensure_files()
        self.user_file.write_text(text.strip() + "\n", encoding="utf-8")

    def today_episode_file(self) -> Path:
        return self.memory_dir / f"{datetime.now():%Y-%m-%d}.md"

    def read_today_episode(self) -> str:
        self.ensure_files()
        path = self.today_episode_file()
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def append_episode(self, text: str):
        self.ensure_files()
        with self.today_episode_file().open("a", encoding="utf-8") as f:
            f.write(f"\n## {datetime.now():%H:%M}\n\n{text.strip()}\n")


MEMORY = MemoryStore(MEMORY_DIR)

COMPACT_PROMPT = """请阅读下面的旧对话，整理出三类信息，并严格按 XML 输出，不要输出任何额外解释。

<episode>
这段时间聊了什么、得出了什么结论（200 字以内，第三人称叙述）
</episode>

<updated_memory>
在【当前长期记忆】基础上合并新信息后的完整长期记忆（Markdown，保留仍然有效的内容，删掉过时的）
</updated_memory>

<updated_user>
在【当前用户画像】基础上合并新信息后的完整用户画像（Markdown）
</updated_user>

【当前长期记忆】
{current_memory}

【当前用户画像】
{current_user}

【今日已有情景记忆】
{today_episode}

【需要整理的旧对话】
{old_conversation}
"""


def messages_to_text(messages: list) -> str:
    """把 history 转成纯文本，喂给模型做摘要"""
    lines = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def build_system_prompt() -> str:
    """每轮都重新拼系统提示，把最新的记忆注入进去"""
    return f"""你是一个乐于助人的助手，用中文简要回答。

【长期记忆 MEMORY.md】
{MEMORY.read_memory()}

【用户画像 USER.md】
{MEMORY.read_user()}

【今日情景记忆】
{MEMORY.read_today_episode() or "(今天还没有压缩出的情景记忆)"}"""


def compact_history(history_messages: list) -> list:
    """history_messages 太长时，把旧消息交给模型摘要成记忆，只保留最近几条原文"""
    if len(history_messages) <= COMPACT_AFTER:
        return history_messages

    old_messages = history_messages[:-RECENT_MESSAGES]
    recent_messages = history_messages[-RECENT_MESSAGES:]
    if not old_messages:
        return history_messages

    prompt = COMPACT_PROMPT.format(
        current_memory=MEMORY.read_memory(),
        current_user=MEMORY.read_user(),
        today_episode=MEMORY.read_today_episode() or "(无)",
        old_conversation=messages_to_text(old_messages),
    )

    messages: list = [
        {"role": "system", "content": "你是记忆整理员。请严格按要求输出 XML，不要输出额外解释。"},
        {"role": "user", "content": prompt},
    ]

    try:
        response = model.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        text = response.choices[0].message.content or ""
    except Exception as e:
        print(f"[记忆压缩失败，保留完整 history_messages]: {e}")
        return history_messages

    episode = extract_tag(text, "episode")
    updated_memory = extract_tag(text, "updated_memory")
    updated_user = extract_tag(text, "updated_user")

    if episode:
        MEMORY.append_episode(episode)
    if updated_memory:
        MEMORY.write_memory(updated_memory)
    if updated_user:
        MEMORY.write_user(updated_user)

    print(f"[记忆已压缩] 旧消息 {len(old_messages)} 条 -> 摘要，保留最近 {len(recent_messages)} 条原文")
    print(f"[写入文件] {MEMORY.memory_file.name} / {MEMORY.user_file.name} / {MEMORY.today_episode_file().name}")
    return recent_messages


history: list = []


def agent(query: str) -> str | None:
    user_message = {"role": "user", "content": query}
    history.append(user_message)
    MEMORY.append_history(user_message)

    messages: list = [{"role": "system", "content": build_system_prompt()}, *history]
    while True:
        response = model.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        reply = response.choices[0].message

        assistant_message = {"role": "assistant", "content": reply.content}
        history.append(assistant_message)
        MEMORY.append_history(assistant_message)

        if not reply.tool_calls:
            answer = reply.content
            break

    # 每轮回答完检查一次，history 过长就压缩成记忆
    history[:] = compact_history(history)
    return answer


if __name__ == '__main__':
    MEMORY.ensure_files()
    print(f"记忆目录：{MEMORY_DIR}")
    print(f"当前 history 超过 {COMPACT_AFTER} 条时自动压缩，保留最近 {RECENT_MESSAGES} 条\n")

    while True:
        user_input = input("用户输入：")
        if user_input == 'exit':
            break
        assistant_output = agent(user_input)
        print("模型回答：", assistant_output)
        print(f"（当前 history {len(history)} 条）\n")
