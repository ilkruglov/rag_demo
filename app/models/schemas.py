"""Pydantic models for request and response payloads."""
from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str
    profile_id: str | None = None


class Source(BaseModel):
    source: str | None = None
    page: int | None = None
    score: float | None = None
    document_title: str | None = None
    document_filename: str | None = None
    section_title: str | None = None
    document_category: str | None = None
    excerpt: str | None = None
    chunk_id: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    profile_id: str | None = None


class DocumentUploadResponse(BaseModel):
    saved_files: list[str]
    ingestion_started: bool
    total_documents: int | None = None
    profile_id: str | None = None


class DocumentInfo(BaseModel):
    file_name: str
    size_bytes: int
    modified_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo] = Field(default_factory=list)
    total_documents: int = 0
    profile_id: str | None = None


class DocumentDeleteRequest(BaseModel):
    file_names: list[str] = Field(default_factory=list)
    profile_id: str | None = None


class DocumentDeleteResponse(BaseModel):
    deleted_files: list[str] = Field(default_factory=list)
    reindexed: bool
    indexes_cleared: bool
    total_documents: int
    profile_id: str | None = None


class ProfileInfo(BaseModel):
    profile_id: str
    label: str
    description: str = ""
    is_active: bool = False
    qdrant_path: str
    storage_dir: str
    raw_documents_dir: str
    qdrant_collection: str
    total_documents: int = 0


class ProfileListResponse(BaseModel):
    active_profile: str
    profiles: list[ProfileInfo] = Field(default_factory=list)
