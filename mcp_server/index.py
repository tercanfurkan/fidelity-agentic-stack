"""Build FAISS index from httpx documentation markdown files."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DOCS_DIR = Path("data/httpx_docs")
FAISS_DIR = Path("data/faiss_index")


def build_index():
    index_file = FAISS_DIR / "index.faiss"
    pkl_file = FAISS_DIR / "index.pkl"

    if index_file.exists() and pkl_file.exists():
        print("FAISS index already exists, skipping rebuild.")
        return

    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {DOCS_DIR}. Run scripts/fetch_docs.py first.")

    print(f"Reading {len(md_files)} markdown files...")
    documents = []
    for f in md_files:
        documents.append(f.read_text(encoding="utf-8"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.create_documents(documents, metadatas=[{"source": f.name} for f in md_files])
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks with text-embedding-3-small...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))
    print(f"Saved FAISS index to {FAISS_DIR}")


if __name__ == "__main__":
    build_index()
