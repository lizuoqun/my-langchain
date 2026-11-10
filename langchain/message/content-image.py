import base64
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
    extra_body={"thinking": {"type": "disabled"}}
)


# 将一张本地图片转换成 Base64 编码的 Data URI 字符串,方便在文本中嵌入图片数据
def encode_image(img_path, img_type='jpeg'):
    with open(img_path, "rb") as img_file:
        return f"data:image/{img_type};base64,{base64.b64encode(img_file.read()).decode("utf-8")}"


img_path = "flower.png"

# 获取图像base64编码字符串
base64_image = encode_image(img_path)

messages = [
    HumanMessage(
        # content=[
        #     {'type': 'text', 'text': '这个图片里面有什么'},
        #     {'type': 'image_url', 'image_url': base64_image}
        # ],
        content_blocks=[
            {'type': 'text', 'text': '这个图片里面有什么'},
            {'type': 'image_url', 'image_url': base64_image}
        ]
    )
]

response = model.invoke(messages)

response.pretty_print()
