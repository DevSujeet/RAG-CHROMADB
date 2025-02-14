from typing import Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLM_Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # env_prefix=CACHE_ENV_PREFIX,  # Match prefix with your .env file #development_
        env_file='.env',
        populate_by_name=True,  # Use field aliases
        extra='ignore',  # Ignore extra inputs from the .env file
        env_file_encoding='utf-8',
    )

    # Match these aliases to .env field keys
    openai_api_key:str = Field(alias='openai_api_key')