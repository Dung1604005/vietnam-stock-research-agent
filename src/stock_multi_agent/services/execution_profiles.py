"""Request-scoped policies; outbound callers must check this guard before I/O."""
from contextlib import contextmanager
from contextvars import ContextVar

from stock_multi_agent.models.execution_profile import ExecutionMode, ExecutionProfile
from stock_multi_agent.models.research_contracts import RequestBudget


class ExternalAccessDenied(RuntimeError):
    pass


_profile = ContextVar('execution_profile', default=ExecutionProfile())


def resolve_profile(mode: ExecutionMode | str | ExecutionProfile = ExecutionMode.STANDARD) -> ExecutionProfile:
    if isinstance(mode, ExecutionProfile):
        if mode.mode == ExecutionMode.OFFLINE and (mode.external_allowed or mode.max_llm_calls != 0):
            raise ValueError('OFFLINE must deny outbound access and LLM calls')
        return mode
    mode = ExecutionMode(str(mode).upper()) if not isinstance(mode, ExecutionMode) else mode
    if mode == ExecutionMode.QUICK:
        return ExecutionProfile(mode=mode, max_llm_calls=0, max_tool_calls=80, deadline_seconds=30,
                                description='Deterministic/cache-first; no LLM generation.')
    if mode == ExecutionMode.DEEP:
        return ExecutionProfile(mode=mode, max_llm_calls=4, max_tool_calls=250, deadline_seconds=180,
                                description='Shared bounded evidence and specialist analysis budget.')
    if mode == ExecutionMode.OFFLINE:
        return ExecutionProfile(mode=mode, max_llm_calls=0, max_tool_calls=250, deadline_seconds=120,
                                external_allowed=False, description='Verified local replay/KB only; no outbound requests.')
    return ExecutionProfile()


def current_profile() -> ExecutionProfile:
    return _profile.get()


@contextmanager
def execution_scope(profile: ExecutionProfile | ExecutionMode | str = ExecutionMode.STANDARD):
    selected = resolve_profile(profile)
    token = _profile.set(selected)
    try:
        yield selected
    finally:
        _profile.reset(token)


def assert_external_allowed(action: str = 'external request') -> None:
    if not current_profile().external_allowed:
        raise ExternalAccessDenied(f'OFFLINE policy blocks {action}')


def constrain_budget(budget: RequestBudget, profile: ExecutionProfile | None = None) -> RequestBudget:
    selected = profile or current_profile()
    changes = {}
    for field in ('max_llm_calls', 'max_tool_calls', 'deadline_seconds'):
        ceiling = getattr(selected, field)
        if ceiling is not None:
            changes[field] = min(getattr(budget, field), ceiling)
    return budget.model_copy(update=changes)
