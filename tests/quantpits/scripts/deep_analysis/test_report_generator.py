import pytest
from datetime import datetime
from quantpits.scripts.deep_analysis.report_generator import ReportGenerator
from quantpits.scripts.deep_analysis.base_agent import AgentFindings, Finding

def test_report_generator_full():
    # Mock findings
    f1 = Finding(severity='critical', category='Market Regime', title='Market Crash', detail='Market is crashing')
    af1 = AgentFindings(
        agent_name='Market Regime',
        window_label='test_window',
        findings=[f1],
        raw_metrics={'regime': 'Bear', 'volatility_20d': 0.3, 'volatility_percentile': 90, 'current_drawdown': 0.1}
    )
    
    f2 = Finding(severity='warning', category='Model Health', title='Model Drift', detail='Model is drifting')
    af2 = AgentFindings(
        agent_name='Model Health',
        window_label='test_window',
        findings=[f2],
        raw_metrics={
            'scorecard': {
                'model_a': {'ic_mean': 0.05, 'ic_latest': 0.04, 'icir_mean': 0.5, 'ic_trend': 'degrading', 'n_snapshots': 10},
                'model_b': {'ic_mean': 0.01, 'ic_latest': None, 'icir_mean': None, 'ic_trend': 'improving', 'n_snapshots': 5}
            },
            'retrain_events': [{'date': '2026-01-01', 'model': 'model_a'}],
            'hyperparams': {'model_a': {'class': 'LGBM', 'lr': 0.1, 'other': 'val'}}
        }
    )
    
    af3 = AgentFindings(
        agent_name='Ensemble Evolution',
        window_label='test_window',
        findings=[],
        raw_metrics={
            'combo_trends': {'combo1': [{'_date': '2026-04-24', 'total_return': 5.0, 'calmar_ratio': 2.0, 'excess_return': 1.0}]},
            'change_events': [
                {'type': 'active_switch', 'date': '2026-01-05', 'detail': 'switched to combo1'},
                {'type': 'composition_change', 'date': '2026-01-06', 'combo': 'combo2'}
            ]
        }
    )
    
    af4 = AgentFindings(
        agent_name='Execution Quality',
        window_label='test_window',
        findings=[],
        raw_metrics={
            'buy_total_friction': 0.005, 'buy_delay_cost': 0.002, 'buy_exec_slippage': 0.003,
            'sell_total_friction': 0.004, 'sell_delay_cost': 0.001, 'sell_exec_slippage': 0.003,
            'explicit_costs': {'fee_ratio': 0.0001},
            'substitution_bias': {'theoretical_substitute_bias_impact': 0.01, 'realized_substitute_bias_impact': 0.005, 'total_missed_count': 2}
        }
    )
    
    af5 = AgentFindings(
        agent_name='Portfolio Risk',
        window_label='test_window',
        findings=[],
        raw_metrics={
            'traditional': {'CAGR_252': 0.2, 'Sharpe': 1.5, 'Max_Drawdown': 0.1, 'Calmar': 2.0, 'Excess_Return_CAGR_252': 0.05},
            'factor_exposure': {'Annualized_Alpha': 0.05, 'Annualized_Alpha_t': 2.5, 'Annualized_Alpha_p': 0.01, 'Beta_Market': 0.9}
        }
    )
    
    af6 = AgentFindings(
        agent_name='Prediction Audit',
        window_label='test_window',
        findings=[],
        raw_metrics={
            'buy_hit_rate': {'overall': {'hit_rate': 0.6, 'n_suggestions': 10, 'avg_return': 0.02}},
            'sell_hit_rate': {'overall': {'hit_rate': 0.7, 'n_suggestions': 5}},
            'consensus_analysis': {'n_high_consensus': 3, 'high_consensus_avg_return': 0.03, 'high_divergence_avg_return': 0.01}
        }
    )
    
    af7 = AgentFindings(
        agent_name='Trade Pattern',
        window_label='test_window',
        findings=[],
        raw_metrics={
            'discipline': {'signal_pct': 80, 'signal_count': 8, 'substitute_pct': 10, 'substitute_count': 1, 'manual_pct': 10, 'manual_count': 1},
            'discipline_score': 85.0
        }
    )
    
    synthesis_result = {
        'health_status': 'Warning',
        'executive_summary_data': {
            'windows_analyzed': ['test_window'],
            'agents_run': ['Agent1', 'Agent2'],
            'critical_count': 1,
            'warning_count': 1,
            'positive_count': 0
        },
        'recommendations': [
            {'priority': 'P0', 'text': 'Fix critical issue', 'source': 'Market Regime'},
            {'priority': 'P1', 'text': 'Check model', 'source': 'Model Health'},
            {'priority': 'P2', 'text': 'Monitor', 'source': 'Portfolio Risk'}
        ],
        'change_impact': [{'event': {'type': 'retrain', 'date': '2026-01-01', 'model': 'model_a'}}],
        'external_notes': 'Some external notes'
    }
    
    generator = ReportGenerator(
        all_findings=[af1, af2, af3, af4, af5, af6, af7],
        synthesis_result=synthesis_result,
        executive_summary="This is the executive summary."
    )
    
    report = generator.generate()
    
    assert "# Deep Analysis Report" in report
    assert "**Health Status:** Warning" in report
    assert "📉 degrading" in report
    assert "📈 improving" in report
    assert "🔄" in report
    assert "🔀" in report

def test_report_generator_empty():
    # Test with no findings to hit "if not findings: return" branches
    generator = ReportGenerator(
        all_findings=[],
        synthesis_result={},
        executive_summary="Summary"
    )
    report = generator.generate()
    assert "1. Market Environment" not in report
    assert "2. Model Health Dashboard" not in report
    assert "3. Ensemble Evolution" not in report
    assert "4. Execution Quality" not in report
    assert "5. Portfolio Risk & Attribution" not in report
    assert "6. Prediction Accuracy Audit" not in report
    assert "7. Trade Behavior" not in report
    assert "8. Holistic Change Impact Assessment" not in report
    assert "9. Prioritized Recommendations" not in report
