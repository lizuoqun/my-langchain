# my-langchain

在项目根目录下创建`.env`文件

## learn 目录结构


```dotenv
DEEPSEEK_API_KEY=sk-crazyThursdayvme50
DEFAULT_API_BASE=https://api.deepseek.com

# 是否启用
LANGSMITH_TRACING=true
# 监控web地址
LANGSMITH_ENDPOINT=https://api.smith.LangChain.com
LANGSMITH_API_KEY=lsv2_pt_crazyThursdayvme50
# 自定义项目名称，可以在Langsmith WebUI监控页面根据名称查看对应的运行记录
LANGSMITH_PROJECT="my-langchain"

# Agent内置工具TavilySearch搜索的key
TAVILY_API_KEY=tvly-dev-crazyThursdayvme50

# PostgreSQL链接配置
POSTGRES_SQL_URL=“postgresql://admin:123456@localhost:5432/langchain_db?sslmode=disable”
```

| 目录 | 说明 |
|------|------|
| [agent](learn/agent) | Agent 创建、调用、流式输出、错误处理、策略与工具集成 |
| [langsmith](learn/langsmith) | LangSmith 追踪与监控集成 |
| [memory](learn/memory) | 会话记忆管理：上下文管理、短期记忆（内存/PostgreSQL）、Store API、工具记忆 |
| [message](learn/message) | 消息处理：图片内容、消息优化、多轮对话、思考过程、ToolMessage |
| [middleware](learn/middleware) | 中间件：Human-in-the-loop、执行顺序、自定义节点、PII 过滤、摘要、TodoList、包装器 |
| [model-batch](learn/model-batch) | 模型批量调用 |
| [model-create](learn/model-create) | 模型创建：类方式与函数方式初始化 |
| [model-ext](learn/model-ext) | 模型扩展功能 |
| [model-invoke](learn/model-invoke) | 模型调用：列表字典、列表对象、响应解析 |
| [model-stream](learn/model-stream) | 模型流式输出 |
| [prompt](learn/prompt) | 提示词模板：调用、特性、初始化、参数类型、模板语法 |
| [rag](learn/rag) |  |
| [structured-output](learn/structured-output) | 结构化输出：Pydantic 模型输出及高级特性 |
| [tools](learn/tools) | 工具定义与调用：装饰器、非装饰器、工具调用模型 |
