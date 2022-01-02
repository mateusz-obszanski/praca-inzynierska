from pathlib import Path
from collections import deque

from rich import print as rprint


def main():
    dir_queue = deque([Path(".")])
    files: deque[Path] = deque()
    to_ignore = {
        ".venv",
        ".ipynb_checkpoints",
        ".pytest_cache",
        ".virtual_documents",
        ".vscode",
        "__pycache__",
        ".git",
    }
    while dir_queue:
        curr = dir_queue.pop()
        if curr.parts and curr.parts[-1] in to_ignore:
            rprint(f"[yellow]ignoring {curr}")
            continue
        rprint(f"[blue]processing dir {curr}")
        for x in curr.iterdir():
            rprint(f"[green]\tprocessing {x}")
            if x.is_dir() and x.parts and x.parts[-1] not in to_ignore:
                dir_queue.append(x)
            elif x.is_file() and x.parts[-1].endswith(".py"):
                files.append(x)
    line_cnt = 0
    for file in files:
        with file.open("r") as f:
            line_cnt += len(f.readlines())

    rprint(f"total number of lines in the project: {line_cnt}")


if __name__ == "__main__":
    main()
