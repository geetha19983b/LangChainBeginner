# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# import wikipedia
# from duckduckgo_search import DDGS
# from mcp.server.fastmcp import FastMCP
# import uvicorn
# import json

# app = FastAPI()
# mcp = FastMCP(name="Tool Server")

# @mcp.tool()
# def wikipedia_search(query: str) -> str:
#     """Search Wikipedia for information"""
#     try:
#         return wikipedia.summary(query, sentences=2)
#     except Exception as e:
#         return f"Error: {str(e)}"

# @mcp.tool()
# def ddg_search(query: str) -> str:
#     """Search DuckDuckGo for information"""
#     try:
#         with DDGS() as ddgs:
#             results = ddgs.text(query, max_results=3)
#             return "\n".join([r["body"] for r in results])
#     except Exception as e:
#         return f"Error: {str(e)}"

# @app.post("/mcp")
# async def mcp_endpoint(request: dict):
#     # Handle MCP requests over HTTP
#     # This is a simplified version - you'd need full MCP HTTP implementation
#     return mcp.handle_request(request)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from mcp.server.fastmcp import FastMCP
import wikipedia
from duckduckgo_search import DDGS
import uvicorn

mcp = FastMCP(name="Tool Server")

@mcp.tool()
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information"""
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def ddg_search(query: str) -> str:
    """Search DuckDuckGo for information"""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            return "\n".join([r["body"] for r in results])
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Get the FastAPI app from FastMCP
    app = mcp.get_fastapi_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)