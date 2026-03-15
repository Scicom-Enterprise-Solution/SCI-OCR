import os


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def to_repo_relative(path: str | None) -> str | None:
    if not path:
        return path
    abs_path = os.path.abspath(path)
    try:
        rel_path = os.path.relpath(abs_path, REPO_ROOT)
    except ValueError:
        return abs_path
    return rel_path.replace(os.sep, "/")


def from_repo_relative(path: str | None) -> str | None:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    normalized = path.replace("\\", os.sep).replace("/", os.sep)
    return os.path.abspath(os.path.join(REPO_ROOT, normalized))
