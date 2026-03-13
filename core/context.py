from contextvars import ContextVar

# Globally accessible request context for logging and tracing
REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="system")
