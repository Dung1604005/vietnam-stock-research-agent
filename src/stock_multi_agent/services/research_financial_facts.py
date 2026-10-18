"""Validate research extraction before it can enter a runtime financial snapshot."""
from __future__ import annotations

from stock_multi_agent.models.data import DataKind, SourceReference, DataQualityIssue, QualitySeverity, QualityStatus
from stock_multi_agent.services.financial_facts import ResearchFinancialFactProposal, validate_research_fact


def apply_research_financial_proposals(data, citations, proposals, ticker):
    updated = data.model_copy(deep=True)
    sources = {item.chunk_id: item for item in citations}
    audit = []
    for raw in proposals[:4]:
        try:
            proposal = ResearchFinancialFactProposal.model_validate(raw)
            citation = sources.get(proposal.source_id)
            if citation is None or citation.quality_tier not in {'T1', 'T2'}:
                raise ValueError('Only retrieved T1/T2 financial evidence can enter numerical analysis')
            if ticker not in citation.tickers or proposal.ticker != ticker:
                raise ValueError('Issuer mismatch')
            if proposal.period_end > data.as_of or citation.published_at > data.fetched_at:
                raise ValueError('Evidence was not available at analysis time')
            item = validate_research_fact(proposal, citation)
            existing = [v for v in updated.financial_statement_observations
                        if v.metric == item.metric and v.period_end == item.period_end
                        and v.frequency == item.frequency]
            if existing:
                # Never pick a restatement/source by list order or overwrite provider history.
                if any((v.value, v.unit, v.currency, v.scale, v.reporting_basis, v.consolidation) !=
                       (item.value, item.unit, item.currency, item.scale, item.reporting_basis, item.consolidation)
                       for v in existing):
                    raise ValueError('Conflicting existing observation requires review')
                audit.append({'source_id': item.source_id, 'metric': item.metric, 'status': 'DUPLICATE'})
                continue
            updated.financial_statement_observations.append(item)
            if not any(ref.source_id == item.source_id for ref in updated.source_refs):
                updated.source_refs.append(SourceReference(provider=f'research:{citation.publisher}',
                    source_id=item.source_id, data_kind={'income_statement': DataKind.INCOME_STATEMENT,
                        'balance_sheet': DataKind.BALANCE_SHEET, 'cash_flow': DataKind.CASH_FLOW}[item.statement],
                    fetched_at=citation.retrieved_at, as_of=item.period_end,
                    source_url=str(citation.source_url), content_sha256=citation.content_sha256))
            audit.append({'source_id': item.source_id, 'metric': item.metric, 'period_end': str(item.period_end),
                          'status': 'ACCEPTED_EXTRACTION', 'semantic_truth_verified': False})
        except (ValueError, TypeError) as exc:
            audit.append({'source_id': raw.get('source_id') if isinstance(raw, dict) else None,
                          'status': 'REJECTED', 'reason': str(exc)[:300]})
    if any(item['status'] == 'ACCEPTED_EXTRACTION' for item in audit):
        if not any(item.code == 'RESEARCH_EXTRACTED_FINANCIAL_FACTS' for item in updated.quality_report.issues):
            updated.quality_report.issues.append(DataQualityIssue(code='RESEARCH_EXTRACTED_FINANCIAL_FACTS',
                severity=QualitySeverity.WARNING,
                message='Exact-source extraction validated mechanically, not independently audited; reduced analytical weight.'))
        updated.quality_report.status = QualityStatus.WARN if updated.quality_report.status == QualityStatus.PASS else updated.quality_report.status
    return updated, audit
