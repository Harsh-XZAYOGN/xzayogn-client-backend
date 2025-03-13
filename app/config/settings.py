import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    JWT_SECRET: str = os.getenv("JWT_SECRET", "secret")
    # careerjet_affid: str
    
    class Config:
        env_file = ".env"
        extra = "ignore"

class JobSearchConfig:
    DEFAULT_PAGESIZE = 6
    MAX_SOURCES = 3

settings = Settings()