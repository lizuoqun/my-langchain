from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

str_chat_template = ChatPromptTemplate.from_messages(
    [
        "Hello, {name}!"
    ]
)

print(str_chat_template.invoke({"name": "LangChain"}))

# --------------------------------------------------------------------------------------------

tuple_chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个翻译大师，你叫{name}"),
        ("human", "帮我翻译你好")
    ]
)
print(tuple_chat_template.invoke({"name": "LangChain"}))

# --------------------------------------------------------------------------------------------

dict_chat_template = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "你是一个翻译大师，你叫{name}"},
        {"role": "human", "content": "帮我翻译你好"}
    ]
)
print(dict_chat_template.invoke({"name": "LangChain"}))

# --------------------------------------------------------------------------------------------

object_chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage("你是一个翻译大师，你叫LangChain"),
        HumanMessage("帮我翻译你好")
    ]
)
print(object_chat_template.invoke({}))

# --------------------------------------------------------------------------------------------

object_ext_chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("你是一个翻译大师，你叫{name}"),
        HumanMessagePromptTemplate.from_template("帮我翻译你好")
    ]
)
print(object_ext_chat_template.invoke({"name": "LangChain"}))

# --------------------------------------------------------------------------------------------

base_chat_template = ChatPromptTemplate.from_messages(
    [
        ChatPromptTemplate.from_messages([
            ("system", "你是一个翻译大师，你叫{name}"),
        ]),
        ChatPromptTemplate.from_messages([
            ("human", "帮我翻译你好")
        ]),
    ]
)
print(base_chat_template.invoke({"name": "LangChain"}))
