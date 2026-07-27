from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有帮助的AI机器人，你的名字是{name}。"),
        ("human", "你好，最近怎么样？"),
        ("ai", "我很好，谢谢！"),
        ("human", "{user_input}"),
    ]
)

invoke_prompt = chat_template.invoke({"name": "Jane", "user_input": "我很好"})

# print(invoke_prompt.to_messages())
# print(type(invoke_prompt))

# --------------------------------------------------------------------------------------------
format_prompt = chat_template.format(name="Jane", user_input="我很好")

# print(format_prompt)
# print(type(format_prompt))

# --------------------------------------------------------------------------------------------
format_message_prompt = chat_template.format_messages(name="Jane", user_input="我很好")
print(format_message_prompt)
print(type(format_message_prompt))
