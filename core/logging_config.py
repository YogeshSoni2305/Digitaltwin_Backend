import logging
import json
import os
from datetime import datetime, timezone

class JsonFormatter(logging.Formatter):
    def format(self, record):
        try:
            # Base log entry with UTC timestamp
            log_entry = {
                "time": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "event": getattr(record, "event", record.msg if isinstance(record.msg, str) else "structured_event"),
                "logger": record.name,
                "process": record.process
            }
            
            # Optional high-priority fields
            for field in ["endpoint", "strategy", "seed", "duration_ms", "error_code"]:
                if hasattr(record, field):
                    log_entry[field] = getattr(record, field)

            # Metadata passed via extra={}
            standard_record_attrs = {
                'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                'funcName', 'levelname', 'levelno', 'lineno', 'module',
                'msecs', 'msg', 'name', 'pathname', 'process', 'processName',
                'relativeCreated', 'stack_info', 'thread', 'threadName'
            }
            
            for key, value in record.__dict__.items():
                if key not in standard_record_attrs and key not in log_entry:
                    log_entry[key] = value

            # Merge dict msg if present
            if isinstance(record.msg, dict):
                log_entry.update(record.msg)

            return json.dumps(log_entry)
        except Exception:
            return f"{datetime.now(timezone.utc).isoformat()} - {record.levelname} - {str(record.msg)}"


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("workforce_twin")
    logger.setLevel(logging.INFO)

    # File Handler
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    # Avoid duplicate logs if re-initialized
    logger.propagate = False
    return logger

# Singleton-ish instance
logger = setup_logging()
