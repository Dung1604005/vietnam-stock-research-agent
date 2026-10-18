from stock_multi_agent.models.research_crawler import DocumentCandidate, IngestionDecision, SourceDefinition, StorageMode


def ingestion_decision(source: SourceDefinition | None, candidate: DocumentCandidate) -> IngestionDecision:
    if source is None or not source.fetch_allowed or not source.accepts(str(candidate.source_url)):
        return IngestionDecision(allowed=False, storage_mode=StorageMode.BLOCKED, reason="SOURCE_NOT_ALLOWED")
    if source.storage_mode == StorageMode.BLOCKED:
        return IngestionDecision(allowed=False, storage_mode=StorageMode.BLOCKED, reason="SOURCE_BLOCKED")
    mode = StorageMode.FULL_TEXT if candidate.approved_full_text else source.storage_mode
    return IngestionDecision(allowed=mode == StorageMode.FULL_TEXT, storage_mode=mode,
                             reason="APPROVED" if mode == StorageMode.FULL_TEXT else "PERMISSION_REQUIRED")
