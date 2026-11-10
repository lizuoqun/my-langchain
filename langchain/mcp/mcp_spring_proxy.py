from mcp import server, types
import httpx

# SpringBoot后端地址
SPRING_BASE_URL = "http://127.0.0.1:8080"

# 新建MCP服务实例
mcp_server = server.Server("springboot-proxy-mcp")

# 定义MCP工具：调用springboot查询订单
@mcp_server.tool()
async def query_order(order_id: str) -> str:
    """
    查询订单信息
    Args:
        order_id: 订单编号
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SPRING_BASE_URL}/api/order/{order_id}")
        resp.raise_for_status()
    return str(resp.json())

# 还可以继续增加更多tool，对应你Spring里其他接口
# @mcp_server.tool()
# async def create_order(xxx):
#    ...调用spring POST接口...

if __name__ == "__main__":
    mcp_server.run()
