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


class PostTradePartialExecutionError(PostTradeExecutionError):
    """Execution failed after some durable outputs may already exist."""

    def __init__(self, message, *, summary, cause):
        super().__init__(message)
        self.summary = summary
        self.cause = cause


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


class PostTradeStateError(PostTradeExecutionError):
    """Base class for deterministic account-state failures."""


class PostTradeStateInputError(PostTradeStateError):
    pass


class SettlementNormalizationError(PostTradeStateInputError):
    pass


class SettlementDateMismatchError(SettlementNormalizationError):
    pass


class UnsupportedSettlementEventError(SettlementNormalizationError):
    pass


class ExecutionReconciliationError(PostTradeStateInputError):
    pass


class PositionNotFoundError(PostTradeStateError):
    pass


class PositionQuantityError(PostTradeStateError):
    pass


class PositionCostError(PostTradeStateError):
    pass


class CashReconciliationError(PostTradeStateError):
    pass


class ValuationMissingError(PostTradeStateError):
    pass


class ValuationSchemaError(PostTradeStateError):
    pass


class PostTradeStateConflictError(PostTradeStateError):
    pass


class PostTradeStatePersistenceError(PostTradeStateError):
    def __init__(self, message, *, committed_outputs=()):
        super().__init__(message)
        self.committed_outputs = tuple(committed_outputs)


class PostTradeTransactionError(PostTradeExecutionError):
    """Base class for recoverable local-filesystem transaction failures."""


class PostTradeTransactionSchemaError(PostTradeTransactionError):
    pass


class PostTradeTransactionConflictError(PostTradeTransactionError):
    pass


class PostTradeTransactionCorruptError(PostTradeTransactionError):
    pass


class PostTradeTransactionRecoveryError(PostTradeTransactionError):
    def __init__(self, message, *, transaction_id=None, committed_outputs=()):
        super().__init__(message)
        self.transaction_id = transaction_id
        self.committed_outputs = tuple(committed_outputs)


class PostTradeConcurrentRunError(PostTradeTransactionError):
    pass


class PostTradeRecoveryRequiredError(PostTradeTransactionError):
    pass


class PostTradeCashflowError(PostTradeExecutionError):
    pass


class PostTradeCashflowConflictError(PostTradeCashflowError):
    pass


class PostTradeReceiptLedgerError(PostTradeExecutionError):
    pass


class PostTradeReceiptLedgerSchemaError(PostTradeReceiptLedgerError):
    pass


class PostTradeAuditError(PostTradeExecutionError):
    pass


class PostTradeClassificationRetryError(PostTradeExecutionError):
    pass
