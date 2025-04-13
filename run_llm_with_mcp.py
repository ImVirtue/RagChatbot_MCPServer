import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import streamlit as st

# Load .env
load_dotenv()

client = OpenAI()
model = os.environ.get("MODEL")

class ConnectionManager:
    def __init__(self, sse_server_map):
        self.sse_server_map = sse_server_map
        self.sessions = {}
        self.exit_stack = AsyncExitStack()

    async def initialize(self):
        for server_name, url in self.sse_server_map.items():
            sse_transport = await self.exit_stack.enter_async_context(sse_client(url=url))
            read, write = sse_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[server_name] = session

    async def list_tools(self):
        tool_map = {}
        consolidated_tools = []
        for server_name, session in self.sessions.items():
            tools = await session.list_tools()
            tool_map.update({tool.name: server_name for tool in tools.tools})
            consolidated_tools.extend(tools.tools)
        return tool_map, consolidated_tools

    async def call_tool(self, tool_name, arguments, tool_map):
        server_name = tool_map.get(tool_name)
        if not server_name:
            print(f"Tool '{tool_name}' not found.")
            return

        session = self.sessions.get(server_name)
        if session:
            result = await session.call_tool(tool_name, arguments=arguments)
            return result.content[0].text

    async def close(self):
        await self.exit_stack.aclose()


# Chat function with OpenAI and tool calling
async def chat(input_messages, tool_map, tools, max_turns=3, connection_manager=None):
    chat_messages = input_messages[:]

    for _ in range(max_turns):
        result = client.chat.completions.create(
            model=model,
            messages=chat_messages,
            tools=tools,
        )

        if result.choices[0].finish_reason == "tool_calls":
            chat_messages.append(result.choices[0].message)

            for tool_call in result.choices[0].message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                server_name = tool_map.get(tool_name, "")

                print(f"\n Tool Call: `{tool_name}` from `{server_name}`")
                print("Arguments:")
                print(json.dumps(tool_args, indent=2))

                observation = await connection_manager.call_tool(
                    tool_name, tool_args, tool_map
                )

                print("\n Tool Observation:")
                print(json.dumps(observation, indent=2))

                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(observation),
                })
        else:
            print("\n Assistant:")
            print(result.choices[0].message.content)
            return result.choices[0].message.content

    # Final response
    result = client.chat.completions.create(
        model=model,
        messages=chat_messages,
    )
    print("\n Final Assistant Response:")
    return str(result.choices[0].message.content)

st.set_page_config(layout="wide", page_title="Chat Application")
col1, col2 = st.columns(2)
with col1:
    st.image(os.path.join(os.getcwd(), "images/hr.png"), width=300)

with col2:
    st.title("""
    :blue[HR Agent With MCP Server]
            Hi I'm HR Assistant
            How can I help you?
    """)

st.markdown(
    """
    <style>
        /* Change sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #696969 !important; /* Blue */
        }

        /* Change all text color inside the sidebar */
        [data-testid="stSidebar"] * {
            color: white !important;
        }

        /* Specifically change sidebar title and subheader color */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2 {
            color: white !important;
        }

        /* Change all buttons background color */
        [data-testid="stButton"] button {
            background-color: 	#181818 !important; /* White background */
            color: white !important; /* Blue text */
            border-radius: 10px !important; /* Rounded corners */
            padding: 10px 20px !important; /* Adjust padding */
            font-size: 16px !important; /* Adjust font size */
        }

        [data-testid="stButton"] button p {
            color: white !important;  /* Ensure text inside button is blue */
        }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.title("📚 Smart Workplace Assistant")
    st.subheader("Features:")
    st.write("- 📄 Preloaded PDF: Workplace Rules and Policies")
    st.write("- 🧠 RAG Chatbot: Ask HR-related questions using GPT-4")
    st.write("- 🔍 PDF Chunking and Semantic Search")
    st.write("- ✉️ Send Emails to Customers via Tool-Calling")
    st.write("- 🧰 Multi-Tool Orchestration with Function Calling")
    st.write("- 🤖 Seamless Integration via MCP for Server-Side Tools")

# Main entry
if __name__ == "__main__":
    sse_server_map = {
        "MCP_SERVER": "http://localhost:8000/sse",
    }

    async def main():
        connection_manager = ConnectionManager(sse_server_map)
        await connection_manager.initialize()

        tool_map, tool_objects = await connection_manager.list_tools()

        tools_json = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tool_objects
        ]
        question = st.chat_input()
        if question:
            st.chat_message("user").write(question)

            input_messages = [
                {"role": "system", "content": "You are a helpful assistant. Use tools to get live data."},
                {"role": "user", "content": question},
            ]

            response = await chat(
                input_messages,
                tool_map,
                tools=tools_json,
                connection_manager=connection_manager,
            )

            st.chat_message("assistant").write(response)
        await connection_manager.close()

    asyncio.run(main())
