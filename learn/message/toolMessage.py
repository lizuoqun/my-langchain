import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
    # 需要禁用推理
    extra_body={"thinking": {"type": "disabled"}}
)


def get_weather(city: str):
    return city + '阳光明媚'


model_with_tools = model.bind_tools([get_weather])

ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"city": "长沙"},
        "id": "call_00_nUD2NC9QRN5Cg1GaoIkBJQ4s"
    }]
)

tool_message = ToolMessage(
    content=get_weather("长沙"),
    tool_call_id="call_00_nUD2NC9QRN5Cg1GaoIkBJQ4s"
)

message = [
    HumanMessage(content="长沙天气如何"),
    ai_message,
    tool_message
]

response = model_with_tools.invoke(message)
response.pretty_print()
