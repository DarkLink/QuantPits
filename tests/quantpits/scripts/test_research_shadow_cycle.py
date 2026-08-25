from quantpits.scripts.research_shadow_cycle import build_parser


def test_b2_cli_requires_all_identity_date_profile_and_output_arguments():
    parser = build_parser()
    args = parser.parse_args([
        "--workspace", "/read-only/workspace", "--qlib-data-dir", "/read-only/qlib",
        "--profile", "/private/profile.json", "--sealed-cycle", "2026-08-21",
        "--anchor", "2026-08-14", "--preferred-start", "2026-07-03",
        "--preferred-end", "2026-08-21", "--top-k", "22",
        "--output-dir", "/tmp/new-b2-root",
    ])
    assert args.anchor == "2026-08-14"
    assert args.top_k == 22
    assert args.output_dir == "/tmp/new-b2-root"
