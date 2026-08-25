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
]
