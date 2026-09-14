from pydantic import BaseModel

from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
from langgraph.runtime import Runtime

load_dotenv(override=True)

model = ChatDeepSeek(
    model='deepseek-v4-flash',
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)


class UserContext(BaseModel):
    username: str
    level: str


class OverAllState(MessagesState):
    input: str
    output: str


def llm_node(state: OverAllState, runtime: Runtime[UserContext]) -> OverAllState:
    runtime_context = runtime.context
    if runtime_context:
        username = runtime_context.username
        level = runtime_context.level
        print(f"\n-------username: {username}, level: {level}----\n")
        if level == 'VIP':
            system_prompt = f"当前用户{username}是VIP，用尊敬的语气回答"
        else:
            system_prompt = f"当前用户{username}不是VIP，简要回答"
    else:
        system_prompt = "直接回答"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["input"])
    ]

    response = model.invoke(messages)

    return {
        "input": state["input"],
        "output": str(response.content),
        "messages": messages + [response]
    }


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node("llm_node", llm_node)  # type: ignore[arg-type]

builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", END)

graph = builder.compile()
res = graph.invoke({"input": "今天是什么日子"}, context=UserContext(username="张三", level="VIP"))
print(res)

print("--" * 50)

res = graph.invoke({"input": "今天是什么日子"}, context=UserContext(username="张三", level=""))
print(res)
