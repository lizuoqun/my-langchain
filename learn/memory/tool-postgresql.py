import os
from typing import NotRequired, cast

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain.tools import tool, ToolRuntime
from langgraph.store.postgres import PostgresStore

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")
POSTGRES_SQL_URL = os.getenv("POSTGRES_SQL_URL")
assert POSTGRES_SQL_URL is not None, "环境变量 POSTGRES_SQL_URL 未配置"

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
    extra_body={"thinking": {"type": "disabled"}}
)


class CustomState(AgentState):
    user_id: NotRequired[str]


@tool(parse_docstring=True)
def save_user_info(name: str, runtime: ToolRuntime) -> str:
    """
    将用户信息保存在长期记忆中

    Args:
        name: 用户名
        runtime: 工具的运行时

    Returns:
        str: 保存状态
    """
    assert runtime.store is not None, "store 未配置"
    runtime.store.put(("users",), runtime.state["user_id"], {"name": name})
    return "saved"


@tool(parse_docstring=True)
def get_user_info(runtime: ToolRuntime) -> str:
    """
    从长期记忆中读取用户信息

    Args:
        runtime: 工具的运行时

    Returns:
        str: 用户信息
    """
    assert runtime.store is not None, "store 未配置"
    item = runtime.store.get(("users",), runtime.state["user_id"])
    return str(item.value) if item else "unknown"


with PostgresStore.from_conn_string(POSTGRES_SQL_URL) as store:
    agent = create_agent(
        model=model,
        tools=[save_user_info, get_user_info],
        store=store,
        system_prompt="用户提及个人信息时及时记录，用户询问个人信息时尝试用工具检索",
        state_schema=cast(type[AgentState], CustomState),
    )

    response_input = agent.invoke({
        "messages": [HumanMessage("你好，很高兴认识你，我是小明")],
        "user_id": "user-1"
    })

    for msg in response_input["messages"]:
        msg.pretty_print()

    print("\n", "-*-" * 100, "\n")

    response_output = agent.invoke({
        "messages": [HumanMessage("我是谁")],
        "user_id": "user-1"
    })

    for msg in response_output["messages"]:
        msg.pretty_print()
