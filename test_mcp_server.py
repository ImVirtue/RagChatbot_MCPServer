from mcp import ClientSession
from mcp.client.sse import sse_client

async def check_mcp_server():
    async with sse_client("http://0.0.0.0:8000/sse") as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()

            # list avai tool
            tools = await session.list_tools()
            print(tools)

            # Call tools
            result = await session.call_tool("add", arguments={"a" : 4, "b" : 6})
            print(result)


if __name__ == '__main__':
    import asyncio
    asyncio.run(check_mcp_server())