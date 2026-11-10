from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore

store = InMemoryStore()

namespace = ("users",)
user_id = 'user-1'
username = "modify"
store.put(namespace, user_id, {"name": username})

store.put(namespace, "user-2", {"name": "张三"})
store.put(namespace, "user-3", {"name": "李四"})
store.put(namespace, "user-4", {"name": "王五"})
store.put(namespace, "user-5", {"name": "张六"})
store.put(("goods",), "goods-1", {"name": "手机"})

print(store.get(namespace, user_id), "\n")

# search检索
for item in store.search(namespace):
    print(item)

print("\n")

for item in store.search(namespace, filter={"name": "张三"}):
    print(item)

print("\n")

DB_URL = "postgresql://admin:123456@localhost:5432/langchain_db?sslmode=disable"
with PostgresStore.from_conn_string(DB_URL) as sqlStore:
    # 创建数据库
    sqlStore.setup()

    sqlStore.put(namespace, user_id, {"name": username})
    print(sqlStore.get(namespace, user_id))
