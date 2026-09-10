"""CLI safe output, explicit output boundaries and truthful failure results."""
import csv
import io
import json
import os
from pathlib import Path

import pytest

from quantpits.scripts import report_forward_window as cli
from quantpits.research import forward_window_report as m
from tests.quantpits.research.test_forward_window_report import (
    fresh_evidence_workspace, fresh_bootstrap_workspace, prepared_inputs, publication,
    continuation, three_window, build,
)


def empty_request(tmp_path):
    roots = {}
    for key in m.ROOTS:
        root = tmp_path / key
        root.mkdir(mode=0o700)
        roots[key] = str(root)
    return dict(roots, schema_version=1, epoch_id='private.epoch', requested_cycles=[dict(
        cycle_index=1, current_cycle_id='2026-09-04', expected_intent_request_digest=None,
        expected_settlement_request_digest=None)])


def test_partial_cli_and_output(tmp_path, capsys):
    request = empty_request(tmp_path)
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(request))
    output = tmp_path / 'output'
    assert cli.main(['--request-file', str(path), '--output-dir', str(output)]) == 2
    stdout = capsys.readouterr().out
    assert request['epoch_id'] not in stdout and str(tmp_path) not in stdout
    assert json.loads(stdout)['written_files'] == ['report.json', 'cycles.csv', 'report.md']
    report = json.loads((output / 'report.json').read_text())
    rows = list(csv.DictReader(io.StringIO((output / 'cycles.csv').read_text())))
    assert len(rows) == 2 and list(rows[0]) == m.CSV_FIELDS
    assert rows[0]['normalized_nav'] == '' and report['rows'][0]['arms'][0]['normalized_nav'] is None
    assert 'NORMALIZATION_UNAVAILABLE' in (output / 'report.md').read_text()
    for path in output.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
    assert cli.main(['--request-file', str(path.parent.parent / 'request.json'), '--output-dir', str(output)]) == 5
    assert json.loads(capsys.readouterr().out)['status'] == 'OUTPUT_FAILED'


@pytest.mark.parametrize('kind', ['source', 'ancestor', 'symlink', 'parent_symlink', 'nonempty', 'missing_parent'])
def test_output_boundaries(tmp_path, kind):
    report = build(empty_request(tmp_path))
    output = tmp_path / 'out'
    if kind == 'source':
        output = Path(report['private_bindings'][m.ROOTS[0]]) / 'out'
    elif kind == 'ancestor':
        output = tmp_path
    elif kind == 'symlink':
        output.symlink_to(report['private_bindings'][m.ROOTS[0]], target_is_directory=True)
    elif kind == 'parent_symlink':
        output.symlink_to(tmp_path, target_is_directory=True)
        output = output / 'child'
    elif kind == 'nonempty':
        output.mkdir(mode=0o700)
        (output / 'keep').write_text('untouched')
    else:
        output = output / 'missing' / 'child'
    with pytest.raises(m.ReportOutputError) as error:
        m.write_report(report, output)
    assert error.value.written_files == []


def test_partial_write_and_interrupt(tmp_path, monkeypatch):
    report = build(empty_request(tmp_path))
    original = m.c4._write_member
    def fail(fd, name, data):
        if name == 'cycles.csv':
            raise OSError('private details')
        original(fd, name, data)
    monkeypatch.setattr(m.c4, '_write_member', fail)
    with pytest.raises(m.ReportOutputError) as error:
        m.write_report(report, tmp_path / 'failed')
    assert error.value.written_files == ['report.json']
    assert (tmp_path / 'failed/report.json').exists()
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(m.c4, '_write_member', interrupt)
    with pytest.raises(KeyboardInterrupt):
        m.write_report(report, tmp_path / 'interrupted')


@pytest.mark.parametrize('data', [b'{"schema_version":1,"schema_version":1}', b'{"x":NaN}', b'{"x":Infinity}', b'[]'])
def test_bad_json_safe(tmp_path, capsys, data):
    path = tmp_path / 'private.json'
    path.write_bytes(data)
    assert cli.main(['--request-file', str(path)]) == 4
    assert str(path) not in capsys.readouterr().out


def test_complete_cli(three_window, tmp_path, capsys):
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(three_window))
    assert cli.main(['--request-file', str(path), '--output-dir', str(tmp_path / 'report')]) == 0
    stdout = capsys.readouterr().out
    assert three_window['epoch_id'] not in stdout and 'SH600' not in stdout and str(tmp_path) not in stdout
    assert json.loads(stdout)['requested_window_chain_verified']
    csv_rows = list(csv.DictReader((tmp_path / 'report/cycles.csv').open()))
    assert len(csv_rows) == 6
    assert [r['role'] for r in csv_rows] == list(m.ROLES) * 3


def test_safe_argument_error(capsys):
    assert cli.main(['--private-unknown', '/private/secret']) == 4
    captured = capsys.readouterr()
    assert 'secret' not in captured.out + captured.err


def test_output_replacement_is_failure(tmp_path, monkeypatch):
    report = build(empty_request(tmp_path))
    target = tmp_path / 'report'
    original = m.c4._write_member
    def replace(fd, name, data):
        original(fd, name, data)
        if name == 'report.md':
            target.rename(tmp_path / 'moved')
            target.mkdir(mode=0o700)
    monkeypatch.setattr(m.c4, '_write_member', replace)
    with pytest.raises(m.ReportOutputError) as exc:
        m.write_report(report, target)
    assert exc.value.attempted_files == ['report.json', 'cycles.csv', 'report.md']
    assert 'report.md' not in exc.value.written_files
