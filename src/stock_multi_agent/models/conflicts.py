from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class ConflictKind(str, Enum):
    MISSING_SPECIALIST_RESULT = "MISSING_SPECIALIST_RESULT"
    SPECIALIST_DISAGREEMENT = "SPECIALIST_DISAGREEMENT"
    TECHNICAL_EVIDENCE_INSUFFICIENT = "TECHNICAL_EVIDENCE_INSUFFICIENT"
    FUNDAMENTAL_EVIDENCE_INSUFFICIENT = "FUNDAMENTAL_EVIDENCE_INSUFFICIENT"
    HIGH_MEASURED_RISK = "HIGH_MEASURED_RISK"
    HIGH_DATA_UNCERTAINTY = "HIGH_DATA_UNCERTAINTY"
    DATA_QUALITY_LIMITATION = "DATA_QUALITY_LIMITATION"
    RISK_VETO = "RISK_VETO"
    LOW_SPECIALIST_CONFIDENCE = "LOW_SPECIALIST_CONFIDENCE"
    HORIZON_MISMATCH = 'HORIZON_MISMATCH'
    METRIC_CONFLICT = 'METRIC_CONFLICT'


class ReanalysisAction(str, Enum):
    NONE = "NONE"
    TECHNICAL = "TECHNICAL"
    RESEARCH_FUNDAMENTAL = "RESEARCH_FUNDAMENTAL"
    DATA_PIPELINE = "DATA_PIPELINE"


class ConflictRecord(BaseModel):
    kind: ConflictKind
    detail: str
    suggested_action: ReanalysisAction = ReanalysisAction.NONE
    dimension: Literal['data', 'time', 'metric', 'interpretation', 'policy'] = 'interpretation'
    severity: Literal['WARNING', 'ERROR'] = 'ERROR'
    resolved: bool = False

    @property
    def blocks_buy(self) -> bool:
        """Interpretive disagreement may coexist with an actionable assessment."""
        return (
            not self.resolved
            and self.severity == 'ERROR'
            and self.dimension in {'data', 'time', 'metric', 'policy'}
        )
