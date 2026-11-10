from typing import Final, Tuple

from langgraph.store.postgres import PostgresStore

POSTGRES_SQL_URL = "postgresql://postgres:123456@127.0.0.1:5432/langchain_db?sslmode=disable"

with PostgresStore.from_conn_string(POSTGRES_SQL_URL) as store:
    store.setup()

    USERS_NAMESPACE: Final[Tuple[str]] = ("users",)
    PREFERENCES_KEY: Final[str] = "preferences"

    namespace1 = (*USERS_NAMESPACE, "张三")
    value1 = {
        "language": "中文",
        "sport": "篮球"
    }

    namespace2 = (*USERS_NAMESPACE, "李四")
    value2 = {
        "language": "英语",
        "sport": "足球"
    }

    store.put(namespace1, PREFERENCES_KEY, value1)
    store.put(namespace2, PREFERENCES_KEY, value2)

    for item in store.search(USERS_NAMESPACE):
        print(item)
