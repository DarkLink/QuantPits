"""Research sidecar primitives, exposed without eager optional dependencies."""

from importlib import import_module


_EXPORT_MODULES = {
    "ReplayContractError": "quantpits.research.replay",
    "ReplayInputError": "quantpits.research.replay",
    "ResearchRankingReplay": "quantpits.research.replay",
    "load_sealed_replay_inputs": "quantpits.research.replay",
    "write_replay_output": "quantpits.research.replay",
    "ExecutionAssumption": "quantpits.research.accounting",
    "ShadowAccountingContractError": "quantpits.research.accounting",
    "ShadowIntentBatch": "quantpits.research.accounting",
    "ShadowPortfolioState": "quantpits.research.accounting",
    "ShadowPortfolioTransition": "quantpits.research.accounting",
    "ShadowQuoteSnapshot": "quantpits.research.accounting",
    "ShadowTransitionResult": "quantpits.research.accounting",
    "AnchorPriceSnapshot": "quantpits.research.intents",
    "CurrentRuleIntentDefinition": "quantpits.research.intents",
    "CurrentRuleShadowIntentPlanner": "quantpits.research.intents",
    "IntentPlanningContractError": "quantpits.research.intents",
    "IntentPlanningParityError": "quantpits.research.intents",
    "IntentPlanningResult": "quantpits.research.intents",
    "ARM_IDS": "quantpits.research.historical_cycle",
    "CashPriceReceipt": "quantpits.research.historical_cycle",
    "HistoricalCycleContractError": "quantpits.research.historical_cycle",
    "HistoricalCycleInputError": "quantpits.research.historical_cycle",
    "HistoricalCycleResult": "quantpits.research.historical_cycle",
    "HistoricalShadowCycleReplay": "quantpits.research.historical_cycle",
    "PublicationReceipt": "quantpits.research.historical_cycle",
    "QlibCashPriceObserver": "quantpits.research.historical_cycle",
    "ReplayProfileReceipt": "quantpits.research.historical_cycle",
    "load_replay_profile": "quantpits.research.historical_cycle",
    "rounding_adjustment_within_bound": "quantpits.research.historical_cycle",
    "write_historical_cycle_output": "quantpits.research.historical_cycle",
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

__all__ = [
    "ReplayContractError",
    "ReplayInputError",
    "ResearchRankingReplay",
    "load_sealed_replay_inputs",
    "write_replay_output",
    "ExecutionAssumption",
    "ShadowAccountingContractError",
    "ShadowIntentBatch",
    "ShadowPortfolioState",
    "ShadowPortfolioTransition",
    "ShadowQuoteSnapshot",
    "ShadowTransitionResult",
    "AnchorPriceSnapshot",
    "CurrentRuleIntentDefinition",
    "CurrentRuleShadowIntentPlanner",
    "IntentPlanningContractError",
    "IntentPlanningParityError",
    "IntentPlanningResult",
    "ARM_IDS",
    "CashPriceReceipt",
    "HistoricalCycleContractError",
    "HistoricalCycleInputError",
    "HistoricalCycleResult",
    "HistoricalShadowCycleReplay",
    "PublicationReceipt",
    "QlibCashPriceObserver",
    "ReplayProfileReceipt",
    "load_replay_profile",
    "rounding_adjustment_within_bound",
    "write_historical_cycle_output",
]
