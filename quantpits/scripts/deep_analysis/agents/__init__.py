"""
Agent registry for the MAS Deep Analysis System.
"""

from .market_regime import MarketRegimeAgent
from .model_health import ModelHealthAgent
from .ensemble_eval import EnsembleEvolutionAgent
from .execution_quality import ExecutionQualityAgent
from .portfolio_risk import PortfolioRiskAgent
from .prediction_audit import PredictionAuditAgent
from .trade_pattern import TradePatternAgent

ALL_AGENTS = {
    'market_regime': MarketRegimeAgent,
    'model_health': ModelHealthAgent,
    'ensemble_eval': EnsembleEvolutionAgent,
    'execution_quality': ExecutionQualityAgent,
    'portfolio_risk': PortfolioRiskAgent,
    'prediction_audit': PredictionAuditAgent,
    'trade_pattern': TradePatternAgent,
}
