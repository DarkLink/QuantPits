"""Private, offline common-window reporting; import/help need no workspace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantpits.research.forward_window_report import (
    parse_request, build_forward_window_report, safe_summary, write_report, ReportOutputError,
)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('CLI_ARGUMENT_INVALID')


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument('--request-file', required=True)
    parser.add_argument('--output-dir')
    try:
        args = parser.parse_args(argv)
        request = parse_request(Path(args.request_file).read_bytes())
        report = build_forward_window_report(**{k: v for k, v in request.items() if k != 'schema_version'})
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        print(json.dumps(dict(status='BLOCKED', reason_codes=['REQUEST_INVALID_OR_UNREADABLE'])))
        return 4
    summary = safe_summary(report)
    if args.output_dir:
        try:
            summary['written_files'] = write_report(report, args.output_dir)
        except ReportOutputError as exc:
            summary.update(status='OUTPUT_FAILED', reason_codes=['OUTPUT_FAILED'], written_files=exc.written_files,
                           attempted_files=exc.attempted_files, output_state='PARTIAL_OR_UNCERTAIN')
            print(json.dumps(summary, sort_keys=True))
            return 5
    print(json.dumps(summary, sort_keys=True))
    return {'COMPLETE': 0, 'PARTIAL': 2, 'BLOCKED': 4}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
