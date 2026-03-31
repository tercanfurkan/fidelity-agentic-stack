"""Download httpx documentation markdown files from GitHub."""

import shutil
import subprocess
from pathlib import Path

DOCS_DIR = Path("data/httpx_docs")


def fetch():
    if list(DOCS_DIR.glob("*.md")):
        print(f"Docs already present in {DOCS_DIR}, skipping download.")
        return

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("/tmp/httpx_repo")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("Cloning httpx repository (shallow)...")
    subprocess.run(
        ["git", "clone", "--depth=1", "https://github.com/encode/httpx.git", str(tmp_dir)],
        check=True,
        capture_output=True,
    )

    source_docs = tmp_dir / "docs"
    count = 0
    for md_file in source_docs.rglob("*.md"):
        dest = DOCS_DIR / md_file.name
        shutil.copy2(md_file, dest)
        count += 1

    shutil.rmtree(tmp_dir)
    print(f"Downloaded {count} markdown files to {DOCS_DIR}")


if __name__ == "__main__":
    fetch()
