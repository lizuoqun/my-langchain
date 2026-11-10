from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有帮助的AI机器人，你的名字是{name}。"),
        ("human", "你好，最近怎么样？"),
        ("ai", "我很好，谢谢！"),
        ("human", "{user_input}"),
    ]
)

prompt = chat_template.invoke({"name": "Alice", "user_input": "你好"})

print(prompt)

# ----------------------------------------------------------------------


chat_template1 = ChatPromptTemplate(
    [
        ("system", "你是一个有帮助的AI机器人，你的名字是{name}。"),
        ("human", "你好，最近怎么样？"),
        ("ai", "我很好，谢谢！"),
        ("human", "{user_input}"),
    ]
)

prompt1 = chat_template1.invoke({"name": "Alice1", "user_input": "你好!"})

print(prompt1)
