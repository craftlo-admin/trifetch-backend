"""
Application Configuration
"""
from typing import Optional
import os


class Settings:
    """Application settings"""
    APP_NAME: str = "FastAPI MVC Application"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Database configuration (for future use)
    DATABASE_URL: Optional[str] = None
    
    # API configuration
    API_PREFIX: str = "/api"
    
    def __init__(self):
        # Load from environment variables if available
        self.APP_NAME = os.getenv("APP_NAME", self.APP_NAME)
        self.APP_VERSION = os.getenv("APP_VERSION", self.APP_VERSION)
        self.DEBUG = os.getenv("DEBUG", str(self.DEBUG)).lower() == "true"
        self.DATABASE_URL = os.getenv("DATABASE_URL", self.DATABASE_URL)
        self.API_PREFIX = os.getenv("API_PREFIX", self.API_PREFIX)


settings = Settings()
