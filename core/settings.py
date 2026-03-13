import os
from pydantic import BaseModel, Field, ValidationError

class AppSettings(BaseModel):
    """
    Validated application settings mapped from environment variables.
    Provides strict bounds for production sizing and safety limits.
    """
    ENV: str = Field(default="development")
    API_KEY: str = Field(default="demo-key-123", min_length=8)
    
    # Resource Limits
    MAX_EMPLOYEES: int = Field(default=5000, le=20000, description="Max org size allowed in engine")
    MAX_TASKS_PER_PROJECT: int = Field(default=500, le=5000, description="Max tasks to prevent memory blowout")
    MAX_MC_ITERATIONS: int = Field(default=50, le=200, description="Cap Monte Carlo loops per request")
    
    # Security / Networking
    TIMEOUT_SECONDS: int = Field(default=30, gt=0, le=120)
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, gt=0)

def load_settings() -> AppSettings:
    try:
        return AppSettings(
            ENV=os.environ.get("ENV", "development"),
            API_KEY=os.environ.get("API_KEY", "demo-key-123"),
            MAX_EMPLOYEES=int(os.environ.get("MAX_EMPLOYEES", 5000)),
            MAX_TASKS_PER_PROJECT=int(os.environ.get("MAX_TASKS_PER_PROJECT", 500)),
            MAX_MC_ITERATIONS=int(os.environ.get("MAX_MC_ITERATIONS", 50)),
            TIMEOUT_SECONDS=int(os.environ.get("TIMEOUT_SECONDS", 30)),
            RATE_LIMIT_PER_MINUTE=int(os.environ.get("RATE_LIMIT_PER_MINUTE", 60)),
        )
    except ValidationError as e:
        import sys
        print(f"CRITICAL: Environment validation failed on startup: {e}")
        sys.exit(1)

settings = load_settings()
