"""Versioned graph proposals. A relationship is a sourced claim, not a fact by fiat."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NodeKind = Literal['Company', 'Ticker', 'Exchange', 'Sector', 'Person', 'Document',
                   'Event', 'Metric', 'Period', 'Topic', 'RiskFactor', 'Claim', 'Evidence']
Relation = Literal['LISTED_AS', 'IN_SECTOR', 'PUBLISHED', 'MENTIONS', 'REPORTS_METRIC',
                   'HAS_EVENT', 'EXPOSED_TO', 'RELATED_TO', 'SUPERSEDES', 'ABOUT', 'SUPPORTED_BY', 'HAS_PERIOD']


class GraphEntity(BaseModel):
    model_config = ConfigDict(extra='forbid')
    entity_id: str = Field(min_length=3, max_length=180)
    kind: NodeKind
    label: str = Field(min_length=1, max_length=500)


class GraphEvidence(BaseModel):
    model_config = ConfigDict(extra='forbid')
    evidence_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_url: str = Field(pattern=r'^https?://')
    quote: str = Field(min_length=1)
    published_at: datetime | None = None
    retrieved_at: datetime
    layer: Literal['theory', 'research', 'structured']

    @field_validator('published_at', 'retrieved_at')
    @classmethod
    def aware(cls, value):
        if value is not None and value.utcoffset() is None:
            raise ValueError('Graph evidence timestamps must be timezone-aware')
        return value


class GraphTriple(BaseModel):
    model_config = ConfigDict(extra='forbid')
    ontology_version: Literal['investment-graph-v2'] = 'investment-graph-v2'
    subject: GraphEntity
    relation: Relation
    object: GraphEntity
    evidence_id: str = Field(min_length=1)
    valid_from: datetime
    valid_to: datetime | None = None
    confidence: float = Field(gt=0, le=1)
    origin: Literal['deterministic', 'llm_proposal'] = 'llm_proposal'

    @model_validator(mode='after')
    def interval(self):
        if self.valid_from.utcoffset() is None or (self.valid_to and self.valid_to.utcoffset() is None):
            raise ValueError('Graph validity timestamps must be timezone-aware')
        if self.valid_to and self.valid_to <= self.valid_from:
            raise ValueError('Invalid graph validity interval')
        allowed = {
            'LISTED_AS': ({'Company'}, {'Ticker'}),
            'IN_SECTOR': ({'Company'}, {'Sector'}),
            'PUBLISHED': ({'Company'}, {'Document'}),
            'REPORTS_METRIC': ({'Document'}, {'Metric'}),
            'HAS_EVENT': ({'Company'}, {'Event'}),
            'EXPOSED_TO': ({'Company', 'Sector'}, {'RiskFactor'}),
            'SUPERSEDES': ({'Document'}, {'Document'}),
            'ABOUT': ({'Claim'}, {'Company', 'Ticker', 'Event', 'Metric'}),
            'SUPPORTED_BY': ({'Claim'}, {'Evidence'}),
            'HAS_PERIOD': ({'Metric', 'Event', 'Claim'}, {'Period'}),
        }
        if self.relation in allowed:
            subjects, objects = allowed[self.relation]
            if self.subject.kind not in subjects or self.object.kind not in objects:
                raise ValueError('Relation incompatible with node types')
        return self
