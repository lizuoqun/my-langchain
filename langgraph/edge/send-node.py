from typing import TypedDict, Literal, Sequence

from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START
from langchain.messages import HumanMessage

from dotenv import load_dotenv
from langgraph.types import Send

load_dotenv(override=True)

model = ChatDeepSeek(
    model='deepseek-v4-flash',
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)


class OverAllState(TypedDict):
    topic: str
    poem: str
    ci_poem: str
    joke: str


class WorkerState(TypedDict):
    """
    私有状态，只对 Worker 节点可用
    """
    content_type: Literal["poem", "ci_poem", "joke"]
    prompt: str


def worker_node(state: WorkerState) -> dict[str, str]:
    content_type = state["content_type"]
    prompt = state["prompt"]

    content = model.invoke([HumanMessage(prompt)]).content
    return {
        content_type: str(content)
    }


def router(state: OverAllState) -> Sequence[Send]:
    prompt = "请生成关于 {} 的 {}"
    english2chinese = {
        "poem": "一首诗",
        "ci_poem": "一首词",
        "joke": "一个笑话"
    }
    return [Send(
        "worker_node",
        {
            "content_type": content_type,
            "prompt": prompt.format(state["topic"], english2chinese[content_type]),
        }
    ) for content_type in ["poem", "ci_poem", "joke"]]


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node(worker_node)  # type: ignore[arg-type]
builder.add_conditional_edges(
    START,
    router,
    path_map=["worker_node"]
)

graph = builder.compile()
res = graph.invoke({"topic": "小猫"})
print(res)

png_bytes = graph.get_graph().draw_mermaid_png()
png_filename = "graph.png"
with open(png_filename, "wb") as f:
    f.write(png_bytes)

raw_mermaid = graph.get_graph().draw_mermaid()
print(raw_mermaid)
