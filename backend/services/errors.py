class ApiError(Exception):
    """An error with an HTTP status, a stable code and a message meant for the API user."""

    def __init__(self, status_code, code, message):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def relative_name(path):
    """Path relative to the project root when possible, for responses that should not expose local paths."""
    from src.config import PROJECT_ROOT

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name
