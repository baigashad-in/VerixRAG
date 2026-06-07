"""
Request validation models.

Never trust raw dicts from request.json(). Pydantic models
validate types, enforce limits, and reject unexpected fields.
This prevents mass assignmnet, type confusion, and parameter
pollution in one step.
"""

from pydantic import BaseModel, Field, field_validator
import re

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    strategy: str = Field(default="none")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v):
        allowed = {"none", "multi", "hyde", "expand"}
        if v not in allowed:
            raise ValueError(f"Strategy must be one of: {allowed}")
        return v

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v):
        # Strip null bytes and control characters
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', v.strip())
    
class IngestRequest(BaseModel):
    directory: str = Field(default="./documents", max_length=200)
    chunk_strategy: str = Field(default="recursive")
    chunk_size: int = Field(default=512, ge=100, le=2000)
    confirm: bool = Field(default=False)
    
    @field_validator("chunk_strategy")
    @classmethod
    def validate_chunk_strategy(cls, v):
        allowed = {"fixed", "recursive", "semantic"}
        if v not in allowed:
            raise ValueError(f"Strategy must be one of: {allowed}")
        return v
    
    @field_validator("directory")
    @classmethod
    def validate_directory(cls, v):
        # Prevent path traversal
        if ".." in v or v.startswith("/"):
            raise ValueError("Invalid directory path")
        return v