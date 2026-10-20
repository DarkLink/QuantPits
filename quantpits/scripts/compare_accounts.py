"""Compare saved actual/paper accounts without activating any workspace."""
import argparse
import json
import sys

from quantpits.research.account_comparison import compare_accounts, load_account_group
from quantpits.research.account_comparison_reports import save_account_comparison


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--group", required=True, help="private YAML/JSON group; relative members resolve beside this file")
    result.add_argument("--since", required=True, help="exact saved start date, YYYY-MM-DD")
    result.add_argument("--date", required=True, help="exact saved end date, YYYY-MM-DD")
    result.add_argument("--output", required=True, help="explicit derived report directory, outside accounts/providers")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        group = load_account_group(args.group)
        report = compare_accounts(group, since=args.since, selected=args.date)
        paths = save_account_comparison(group, report, args.output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "members": [{"id": m["id"], "status": m["status"]} for m in report["members"]],
                      "reports": paths}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
