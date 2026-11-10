from typing import Literal, Final, Tuple

from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage

from dotenv import load_dotenv
from langgraph.runtime import Runtime
from langgraph.store.postgres import PostgresStore

load_dotenv(override=True)

model = ChatDeepSeek(
    model='deepseek-v4-flash',
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)


class OverAllState(MessagesState):
    username: str
    user_input: str
    output: str
    preferences: dict[str, str]


def router(state: OverAllState) -> Literal["check_preferences_node", "llm_node"]:
    print(f"router preferences---,{state.get('preferences')}\n\n")

    if not state.get("preferences"):
        return "check_preferences_node"

    return "llm_node"


USERS_NAMESPACE: Final[Tuple[str]] = ("users",)
PREFERENCES_KEY: Final[str] = "preferences"


def check_preferences_node(state: OverAllState, runtime: Runtime) -> dict[str, str]:
    namespace = (*USERS_NAMESPACE, state["username"])
    store = runtime.store
    item = store.get(namespace, PREFERENCES_KEY)
    print(f"check_preferences_node ----- namespace: {namespace}, item: {item}\n\n")
    if item:
        return {
            "preferences": item.value
        }
    else:
        return {}


def llm_node(state: OverAllState) -> dict[str, str]:
    content = f"这是用户的偏好：{state.get("preferences", {})}，\n这是用户的需求：{state["user_input"]}"
    response = model.invoke([HumanMessage(content=content)])

    return {
        "output": str(response.content)
    }


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node("check_preferences_node", check_preferences_node)  # type: ignore[arg-type]
builder.add_node("llm_node", llm_node)  # type: ignore[arg-type]

builder.add_conditional_edges(START, router,
                              path_map={"check_preferences_node": "check_preferences_node", "llm_node": "llm_node"})
builder.add_edge("check_preferences_node", "llm_node")
builder.add_edge("llm_node", END)

POSTGRES_SQL_URL = "postgresql://postgres:123456@127.0.0.1:5432/langchain_db?sslmode=disable"
with PostgresSaver.from_conn_string(POSTGRES_SQL_URL) as checkpointer, PostgresStore.from_conn_string(
        POSTGRES_SQL_URL) as store:
    checkpointer.setup()
    store.setup()

    graph = builder.compile(checkpointer=checkpointer, store=store)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-007"
        }
    }

    res = graph.invoke({"username": "张三", "user_input": "我说什么语言、我喜欢什么运动"}, config=config)
    print(res)

    png_bytes = graph.get_graph().draw_mermaid_png()
    raw_mermaid = graph.get_graph().draw_mermaid()
    print(raw_mermaid)
