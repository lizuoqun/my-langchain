from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，目标用户是{audience}"),
    ("user", "{task}")
])
# 部分变量预填充 partial
custom_template = template.partial(role="手机销售", audience="张三")

print(custom_template.format_messages(task="这个咋卖？"))

# -------------------------------------------------------------------------------

# 消息占位符
placeholder_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有用的AI助手"),
        ("placeholder", "{conversation}"),
        MessagesPlaceholder("conversation")
    ]
)

prompt_value = placeholder_template.invoke({
    "conversation": [
        HumanMessage("你好"),
        AIMessage("你好"),
        HumanMessage("1+1=?"),
        AIMessage("2"),
        HumanMessage("我刚才问了什么问题？"),
        AIMessage("你刚才问的是1+1=？")
    ]
})

print(prompt_value)
