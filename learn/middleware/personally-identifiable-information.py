import os
import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE
)


# 自定义检测函数
def detect_phone_number(content: str):
    return [
        {
            "text": m.group(0),  # 提取出具体匹配到的 11 位数字文本（例如"13800138000"）
            "start": m.start(),  # 这段数字在原文本中的“起始索引位置”（从 0 开始算）
            "end": m.end()  # 这段数字在原文本中的“结束索引位置”
        }
        for m in re.finditer(r"[0-9]{11}", content)
    ]


agent = create_agent(
    model,
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware("url", strategy="hash", apply_to_input=True),
        PIIMiddleware("mac_address", strategy="mask", apply_to_input=True),
        PIIMiddleware("ip", strategy="block", apply_to_input=True),
        PIIMiddleware("api_key", strategy="hash", apply_to_input=True, detector=r"sk-[a-zA-Z0-9]+"),
        PIIMiddleware("phone_number", strategy="mask", apply_to_input=True, detector=detect_phone_number)

    ]
)
try:
    response = agent.invoke({
        "messages": [
            # HumanMessage("帮我向 156168188@qq.com 发送一封邮件，同时查看银行卡号： 5105-1051-0510-5100 的余额，"
            #              "访问 https://localhost:12345，确认这是不是 MAC地址： 11-11-11-11-11-11"),
            # HumanMessage("看看这个 IP 能不能 ping 通：192.168.10.1")
            HumanMessage("试一试我这个sk-crazyThursdayvme50是否有效，给张三打个电话，他的手机号是：13800138000")
        ]
    })
    for msg in response["messages"]:
        msg.pretty_print()
except Exception as e:
    print(f"检测到IP，抛出异常：{e}")
