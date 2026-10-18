"""Training and adapter contracts. Approval is recorded, never inferred from loss."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TrainingMessage(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: Literal['system', 'user', 'assistant']
    content: str = Field(min_length=1, max_length=32000)


class FineTuningExample(BaseModel):
    model_config = ConfigDict(extra='forbid')
    example_id: str = Field(min_length=1, max_length=120)
    task: Literal['nlu_routing', 'financial_extraction', 'grounded_synthesis']
    messages: list[TrainingMessage] = Field(min_length=1, max_length=20)
    approved_output: dict[str, Any]
    output_schema: dict[str, Any]
    schema_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    issuer_group: str = ''
    template_group: str = ''
    document_family: str = ''
    conversation_group: str = ''
    available_at: datetime | None = None
    source_ids: list[str] = Field(default_factory=list)
    source_checksums: list[str] = Field(default_factory=list)
    usage_rights: Literal['AUTHORED', 'PERMISSION_GRANTED', 'PUBLIC_DOMAIN', 'UNVERIFIED'] = 'UNVERIFIED'
    permission_reference: str = ''
    consent_to_training: bool = False
    cloud_upload_allowed: bool = False
    contains_personal_portfolio: bool = False
    label_origin: Literal['human_reviewed', 'synthetic_draft', 'unreviewed_snapshot'] = 'synthetic_draft'
    approval_status: Literal['DRAFT', 'APPROVED', 'REJECTED'] = 'DRAFT'
    reviewer: str = ''
    reviewed_at: datetime | None = None

    @field_validator('available_at', 'reviewed_at')
    @classmethod
    def aware_dates(cls, value):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError('Training dates must contain an explicit timezone')
        return value

    @model_validator(mode='after')
    def reviewed_labels_are_explicit(self):
        if self.approval_status == 'APPROVED' and (
            self.label_origin != 'human_reviewed' or not self.reviewer.strip() or self.reviewed_at is None
        ):
            raise ValueError('Approval requires a named human reviewer, timestamp and human_reviewed label')
        if self.usage_rights == 'PERMISSION_GRANTED' and not self.permission_reference.strip():
            raise ValueError('Permission must have a reviewable reference')
        if self.messages[-1].role != 'user':
            raise ValueError('Input messages must end with user; approved output is separate')
        return self


class DatasetManifest(BaseModel):
    schema_version: str = 'llm-training-dataset-v1'
    status: Literal['REVIEWED_TRAINING_SET', 'SYNTHETIC_DEMO_ONLY']
    created_at: datetime
    task: str
    dataset_digest: str
    split_seed: int
    split_hashes: dict[str, str]
    split_counts: dict[str, int]
    group_counts: dict[str, int]
    grouping_dimensions: list[str]
    overlap: dict[str, int]
    rejected_count: int
    deduplicated_count: int
    cloud_upload_authorized: bool = False
    promotion_eligible: bool = False
    schema_hash: str
    prompt_hash: str
    limitations: list[str] = Field(default_factory=list)


class AdapterManifest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal['llm-adapter-v1'] = 'llm-adapter-v1'
    adapter_id: str = Field(min_length=1, max_length=100)
    base_model: str = 'Qwen/Qwen3-4B-Instruct-2507'
    base_revision: str = Field(pattern=r'^[a-f0-9]{40}$')
    tokenizer_revision: str = Field(pattern=r'^[a-f0-9]{40}$')
    chat_template_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    prompt_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    output_schema_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    dataset_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    dataset_status: Literal['REVIEWED_TRAINING_SET', 'SYNTHETIC_DEMO_ONLY']
    task: Literal['nlu_routing', 'financial_extraction', 'grounded_synthesis']
    adapter_files: dict[str, str] = Field(min_length=1)
    quantization: Literal['nf4-fp16'] = 'nf4-fp16'
    library_versions: dict[str, str] = Field(default_factory=dict)
    training_config: dict[str, Any] = Field(default_factory=dict)
    license: Literal['apache-2.0'] = 'apache-2.0'
    license_reviewed: bool = False
    evaluation_status: Literal['PENDING', 'REJECTED', 'INCONCLUSIVE', 'PASSED'] = 'PENDING'
    evaluation_report: str = ''
    evaluation_sha256: str = ''
    approval_status: Literal['PENDING', 'APPROVED', 'REJECTED'] = 'PENDING'
    reviewer: str = ''
    reviewed_at: datetime | None = None

    @property
    def identity(self) -> str:
        payload = self.model_dump(mode='json')
        return self.adapter_id + ':' + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
