"""FastMCP server exposing httpx documentation retrieval via FAISS."""

from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

load_dotenv()

mcp = FastMCP("httpx-retriever")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    "data/faiss_index", embeddings, allow_dangerous_deserialization=True
)


@mcp.tool()
def retrieve(query: str, k: int = 3) -> str:
    """Retrieve top-k relevant chunks from httpx documentation."""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    mcp.run(transport="sse", host="localhost", port=8765)
