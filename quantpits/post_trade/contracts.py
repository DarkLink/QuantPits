"""Typed contracts shared by post-trade commands and broker adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

PostTradeScope = Literal["all", "state", "execution"]
PostTradeStream = Literal["settlement", "order", "trade"]
SourceStatus = Literal["present", "missing", "assumed_empty", "already_ingested", "changed"]
IssueSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class PostTradeSourceRef:
    stream: PostTradeStream
    trade_date: str
    path: Path
    display_path: str
    status: SourceStatus
    fingerprint: Optional[str] = None
    size_bytes: Optional[int] = None


@dataclass(frozen=True)
class PostTradeIntakeIssue:
    code: str
    severity: IssueSeverity
    message: str
    stream: Optional[PostTradeStream] = None
    trade_date: Optional[str] = None


@dataclass(frozen=True)
class PostTradeInputCatalog:
    date_from: str
    date_to: str
    settlement_sources: Tuple[PostTradeSourceRef, ...] = ()
    order_sources: Tuple[PostTradeSourceRef, ...] = ()
    trade_sources: Tuple[PostTradeSourceRef, ...] = ()
    issues: Tuple[PostTradeIntakeIssue, ...] = ()

    def sources_for(self, stream: PostTradeStream) -> Tuple[PostTradeSourceRef, ...]:
        return getattr(self, "%s_sources" % stream)

    def source_for_date(self, stream: PostTradeStream, trade_date: str) -> Optional[PostTradeSourceRef]:
        return next((item for item in self.sources_for(stream) if item.trade_date == trade_date), None)

    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class ParsedPostTradeInput:
    source: PostTradeSourceRef
    dataframe: object
    row_count: int


class PostTradePlanError(ValueError):
    pass


class PostTradeExecutionError(RuntimeError):
    pass


class PostTradeInputError(PostTradeExecutionError):
    pass


class PostTradeInputMissingError(PostTradeInputError):
    pass


class BrokerParseError(PostTradeInputError):
    pass


class BrokerSchemaError(PostTradeInputError):
    pass


class SourceChangedError(PostTradeInputError):
    pass


class ExecutionEvidenceGapError(PostTradeInputError):
    pass


class IngestionPersistenceError(PostTradeExecutionError):
    pass
