
from mcp.server.fastmcp import FastMCP
from send_mail import send_email as raw_send_email
from rag_pipeline import ask_from_pdf as raw_ask_from_pdf

# Auto open in port 8000
mcp = FastMCP(
    name="mcp-server",
)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def get_current_temperature_by_city(city_name: str) -> str:
    """Get current temperature of a city"""
    return "20 degrees celcius"

@mcp.tool()
def send_email(receiver: str, subject: str, body: str) -> str:
    """Send an email to a given recipient with a subject and message"""
    return raw_send_email(receiver, subject, body)

@mcp.tool()
def  ask_from_pdf(question: str) -> str:
    """
       Query and answer questions from a preloaded PDF document
       Default: The PDF file is already loaded and indexed — no need to upload anything.

       Parameters:
           question (str): The question asked by the user.

       Returns:
           str: The answer generated using retrieved context from the PDF.
       """
    return raw_ask_from_pdf(question)

if __name__ == "__main__":
    print("Listening")
    mcp.run(transport='sse')


