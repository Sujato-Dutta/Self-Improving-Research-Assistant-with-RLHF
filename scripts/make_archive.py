import os
import tarfile
from pathlib import Path

exclude_dirs = {"venv", ".git", ".pytest_cache", "__pycache__", ".tempmediaStorage"}
exclude_files = {"project_transfer.tar.gz"}

root_dir = Path(__file__).resolve().parent.parent
archive_path = root_dir / "project_transfer.tar.gz"

with tarfile.open(archive_path, "w:gz") as tar:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".")]
        
        for f in filenames:
            if f in exclude_files or f.endswith(".pyc") or f.startswith("."):
                continue
            full_path = os.path.join(dirpath, f)
            arcname = os.path.relpath(full_path, root_dir).replace("\\", "/")
            tar.add(full_path, arcname=arcname)

print(f"Created archive {archive_path}, size: {os.path.getsize(archive_path)} bytes")
