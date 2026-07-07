import pytest
from quantpits.scripts.deep_analysis.synthesizer import Synthesizer
from quantpits.scripts.deep_analysis.base_agent import AgentFindings, Finding

def test_synthesizer_cross_reference_alpha_decay():
    # Case: Model IC declining + Market regime high-vol
    f_model = Finding(severity='warning', category='Model Health', 
                      title='Model IC declining', detail='...')
    af_model = AgentFindings(agent_name='Model Health', window_label='1m', 
                             findings=[f_model])
    
    af_market = AgentFindings(agent_name='Market Regime', window_label='1m', 
                              raw_metrics={'regime': 'High-Vol'})
    
    synth = Synthesizer([af_model, af_market])
    result = synth.synthesize()
    
    # Should find the cross-finding
    cross = [f for f in result['cross_findings'] if 'IC decline' in f.title]
    assert len(cross) == 1
    assert cross[0].severity == 'warning'

def test_synthesizer_health_status():
    # Healthy case
    synth_ok = Synthesizer([])
    assert "HEALTHY" in synth_ok.synthesize()['health_status']
    
    # Critical case
    f_crit = Finding(severity='critical', category='Test', title='T', detail='D')
    af_crit = AgentFindings(agent_name='A1', window_label='1m', findings=[f_crit, f_crit])
    synth_crit = Synthesizer([af_crit])
    assert "CRITICAL" in synth_crit.synthesize()['health_status']

def test_synthesizer_recommendations():
    f = Finding(severity='critical', category='Cross-Agent', title='T', detail='D1')
    af = AgentFindings(agent_name='A1', window_label='1m', recommendations=['R1'])
    
    # Simulate a cross-finding by manually adding it or triggering a rule
    # Here we just check if it collects recommendations from agents
    synth = Synthesizer([af])
    result = synth.synthesize()
    
    assert any(r['text'] == 'R1' for r in result['recommendations'])


# ===================================================================
# Coverage gap tests
# ===================================================================

def test_health_status_alert():
    """Line 334: exactly 1 critical → ALERT status (not CRITICAL)."""
    f1 = Finding(severity='critical', category='Test', title='T1', detail='D1')
    af = AgentFindings(agent_name='A1', window_label='1m', findings=[f1])
    synth = Synthesizer([af])
    status = synth.synthesize()['health_status']
    assert "ALERT" in status


def test_change_impact_with_ensemble_events(tmp_path):
    """Lines 233-266: change_events from Ensemble Evolution → impact assessment."""
    af_ensemble = AgentFindings(
        agent_name='Ensemble Evolution', window_label='1m',
        raw_metrics={'change_events': [
            {'type': 'member_replaced', 'combo': 'c1', 'date': '2026-05-01',
             'old_model': 'm1', 'new_model': 'm2'},
        ]},
    )
    af_port = AgentFindings(
        agent_name='Portfolio Risk', window_label='1m',
        raw_metrics={'traditional': {'Sharpe': 2.0, 'CAGR_252': 0.15, 'Max_Drawdown': 0.1}},
    )
    synth = Synthesizer([af_ensemble, af_port])
    # _assess_change_impact is called during synthesize
    result = synth.synthesize()
    # Should not crash and should produce recommendations
    assert 'health_status' in result


def test_change_impact_with_retrain_events(tmp_path):
    """Lines 236-241: retrain_events from Model Health → change_events entries."""
    af_model = AgentFindings(
        agent_name='Model Health', window_label='1m',
        raw_metrics={'retrain_events': [
            {'model': 'm1', 'date': '2026-05-01'},
        ]},
    )
    af_port = AgentFindings(
        agent_name='Portfolio Risk', window_label='1m',
        raw_metrics={'traditional': {'Sharpe': 1.8, 'CAGR_252': 0.12, 'Max_Drawdown': 0.15}},
    )
    synth = Synthesizer([af_model, af_port])
    result = synth.synthesize()
    assert 'health_status' in result

    # The retrain event should create a change event
    impact = synth._assess_change_impact()
    assert len(impact) >= 1
    assert any(e['event']['type'] == 'retrain' for e in impact)


def test_change_impact_no_events(tmp_path):
    """Line 243-244: no change_events → returns empty impact."""
    af = AgentFindings(agent_name='Ensemble Evolution', window_label='1m',
                       raw_metrics={})
    synth = Synthesizer([af])
    impact = synth._assess_change_impact()
    assert impact == []


def test_alpha_significance_all_insignificant(tmp_path):
    """Lines 200-201: all Portfolio Risk windows have p>0.1 → cross-finding."""
    af1 = AgentFindings(
        agent_name='Portfolio Risk', window_label='1m',
        raw_metrics={'factor_exposure': {'Annualized_Alpha_p': 0.5}},
    )
    af2 = AgentFindings(
        agent_name='Portfolio Risk', window_label='3m',
        raw_metrics={'factor_exposure': {'Annualized_Alpha_p': 0.3}},
    )
    synth = Synthesizer([af1, af2])
    cross = synth._check_alpha_significance()
    # Both windows have p > 0.1 → all_insignificant → finding generated
    assert len(cross) == 1
    assert 'statistically significant' in cross[0].title.lower()


def test_synthesizer_time_horizon_reversal():
    af_3m = AgentFindings(
        agent_name='Ensemble Evolution', window_label='3m',
        raw_metrics={'best_combo': {'name': 'Defensive_V2', 'excess_return': 0.15}}
    )
    af_1y = AgentFindings(
        agent_name='Ensemble Evolution', window_label='1y',
        raw_metrics={'best_combo': {'name': 'Defensive_V2', 'excess_return': -0.08}}
    )
    synth = Synthesizer([af_3m, af_1y])
    result = synth.synthesize()
    
    cross = [f for f in result['cross_findings'] if 'OOS Time Horizon Reversal' in f.title]
    assert len(cross) == 1
    assert cross[0].severity == 'warning'


def test_synthesizer_ic_combo_contradiction():
    f_model = Finding(severity='info', category='Model Health', title='Model IC', detail='IC trend is improving')
    af_model = AgentFindings(agent_name='Model Health', window_label='1m', findings=[f_model])
    
    f_ee = Finding(severity='warning', category='Ensemble Evolution', title='Defensive_V2: Calmar degrading', detail='...')
    af_ee = AgentFindings(agent_name='Ensemble Evolution', window_label='1m', findings=[f_ee])
    
    synth = Synthesizer([af_model, af_ee])
    result = synth.synthesize()
    
    cross = [f for f in result['cross_findings'] if 'IC-Combo Performance Contradiction' in f.title]
    assert len(cross) == 1
    assert cross[0].severity == 'warning'
