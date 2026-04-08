import pytest
import os
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from quantpits.scripts import migrate_ensemble_outputs

def test_extract_date():
    assert migrate_ensemble_outputs._extract_date("brute_force_results_2023-10-01.csv") == "2023-10-01"
    assert migrate_ensemble_outputs._extract_date("minentropy_2022-01-15.json") == "2022-01-15"
    assert migrate_ensemble_outputs._extract_date("no_date.txt") is None
    assert migrate_ensemble_outputs._extract_date("file_with_2023-99-99_invalid but matches regex.txt") == "2023-99-99"

def test_detect_script_from_dir():
    assert migrate_ensemble_outputs._detect_script_from_dir("output/brute_force") == "brute_force"
    assert migrate_ensemble_outputs._detect_script_from_dir("output/brute_force_fast/") == "brute_force_fast"
    assert migrate_ensemble_outputs._detect_script_from_dir("other/random_dir") is None

def test_detect_script_from_metadata(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    
    # Test run_metadata
    meta_path = source_dir / "run_metadata_2023-10-01.json"
    with open(meta_path, "w") as f:
        json.dump({"script_used": "brute_force_ensemble"}, f)
    
    assert migrate_ensemble_outputs._detect_script_from_metadata(str(source_dir), "2023-10-01") == "brute_force"
    
    # Test minentropy_metadata
    meta_path_me = source_dir / "minentropy_metadata_2023-10-02.json"
    with open(meta_path_me, "w") as f:
        json.dump({"script_used": "minentropy"}, f)
    
    assert migrate_ensemble_outputs._detect_script_from_metadata(str(source_dir), "2023-10-02") == "minentropy"
    
    # Test non-existent metadata
    assert migrate_ensemble_outputs._detect_script_from_metadata(str(source_dir), "2023-10-03") is None

def test_plan_migration(tmp_path):
    source_dir = tmp_path / "output/brute_force"
    source_dir.mkdir(parents=True)
    target_base = tmp_path / "output/ensemble_runs"
    
    # Create mock files
    files = [
        "brute_force_results_2023-10-01.csv",
        "oos_report_2023-10-01.txt",
        "run_metadata_2023-10-01.json",
        "unknown_file_2023-10-01.txt",
        "minentropy_results_2023-10-02.csv"
    ]
    for f in files:
        (source_dir / f).write_text("dummy content")
        
    plan = migrate_ensemble_outputs.plan_migration(str(source_dir), str(target_base))
    
    # Verify plan length (5 files)
    assert len(plan) == 5
    
    # Check specific mappings
    plan_dict = {os.path.basename(p[0]): p for p in plan}
    
    # brute_force_results_... -> brute_force_2023-10-01/is/results.csv
    src, dst, date, script = plan_dict["brute_force_results_2023-10-01.csv"]
    assert date == "2023-10-01"
    assert script == "brute_force"
    assert dst.endswith("brute_force_2023-10-01/is/results.csv")
    
    # oos_report_... -> brute_force_2023-10-01/oos/oos_report.txt
    src, dst, date, script = plan_dict["oos_report_2023-10-01.txt"]
    assert dst.endswith("brute_force_2023-10-01/oos/oos_report.txt")
    
    # run_metadata_... -> brute_force_2023-10-01/run_metadata.json
    src, dst, date, script = plan_dict["run_metadata_2023-10-01.json"]
    assert dst.endswith("brute_force_2023-10-01/run_metadata.json")
    
    # unknown_file_... -> brute_force_2023-10-01/is/unknown_file_2023-10-01.txt
    src, dst, date, script = plan_dict["unknown_file_2023-10-01.txt"]
    assert dst.endswith("brute_force_2023-10-01/is/unknown_file_2023-10-01.txt")

    # minentropy_results_... -> minentropy_2023-10-02/is/results.csv
    src, dst, date, script = plan_dict["minentropy_results_2023-10-02.csv"]
    assert script == "minentropy"
    assert dst.endswith("minentropy_2023-10-02/is/results.csv")

def test_execute_migration(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    target_base = tmp_path / "target"
    
    src_file = source_dir / "test_2023-10-01.csv"
    src_file.write_text("content")
    
    dst_file = target_base / "run_2023-10-01/is/results.csv"
    
    plan = [(str(src_file), str(dst_file), "2023-10-01", "run")]
    
    # Test dry run
    migrate_ensemble_outputs.execute_migration(plan, dry_run=True)
    assert not dst_file.exists()
    
    # Test actual execution
    migrate_ensemble_outputs.execute_migration(plan, dry_run=False)
    assert dst_file.exists()
    assert dst_file.read_text() == "content"
    
    # Test already exists (should not crash)
    migrate_ensemble_outputs.execute_migration(plan, dry_run=False)
    assert dst_file.exists()

def test_main(tmp_path, monkeypatch, capsys):
    source_dir = tmp_path / "output/brute_force"
    source_dir.mkdir(parents=True)
    target_dir = tmp_path / "output/ensemble_runs"
    
    (source_dir / "results_2023-10-01.csv").write_text("data")
    
    import sys
    monkeypatch.setattr(sys, 'argv', [
        "script.py", 
        "--source", str(source_dir), 
        "--target", str(target_dir),
        "--dry-run"
    ])
    
    migrate_ensemble_outputs.main()
    captured = capsys.readouterr()
    
    assert "Ensemble Output Migration Tool" in captured.out
    assert "DRY RUN 模式" in captured.out
    assert not target_dir.exists()

def test_plan_migration_invalid_dir(capsys):
    plan = migrate_ensemble_outputs.plan_migration("/non/existent/dir", "target")
    assert plan == []
    captured = capsys.readouterr()
    assert "源目录不存在" in captured.out

def test_plan_migration_edge_cases(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    
    # 1. Subdirectory (should be skipped by isfile check)
    (source_dir / "subdir").mkdir()
    
    # 2. File without date (should be skipped)
    (source_dir / "no_date.txt").write_text("data")
    
    # 3. Non-matched file with date (should go to 'is/' with original name)
    (source_dir / "random_2023-10-01.txt").write_text("data")
    
    plan = migrate_ensemble_outputs.plan_migration(str(source_dir), "target")
    
    # Only the non-matched file with date should be in the plan
    assert len(plan) == 1
    src, dst, date, script = plan[0]
    assert os.path.basename(src) == "random_2023-10-01.txt"
    assert "target/unknown_2023-10-01/is/random_2023-10-01.txt" in dst.replace("\\", "/")

def test_plan_migration_minentropy_prefix(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    
    # Unmatched file with minentropy_ prefix
    (source_dir / "minentropy_unknown_2023-10-01.txt").write_text("data")
    
    # Also test date_to_script with metadata
    meta_path = source_dir / "run_metadata_2023-10-02.json"
    with open(meta_path, "w") as f:
        json.dump({"script_used": "brute_force_ensemble"}, f)
    (source_dir / "some_file_2023-10-02.txt").write_text("data")
    
    plan = migrate_ensemble_outputs.plan_migration(str(source_dir), "target")
    plan_dict = {os.path.basename(p[0]): p for p in plan}
    
    # Check minentropy prefix handling in unmatched
    src, dst, date, script = plan_dict["minentropy_unknown_2023-10-01.txt"]
    assert script == "minentropy"
    assert "target/minentropy_2023-10-01/is/minentropy_unknown_2023-10-01.txt" in dst.replace("\\", "/")
    
    # Check date_to_script usage
    src, dst, date, script = plan_dict["some_file_2023-10-02.txt"]
    assert script == "brute_force"

def test_execute_migration_empty_plan(capsys):
    migrate_ensemble_outputs.execute_migration([])
    captured = capsys.readouterr()
    assert "无文件需要迁移" in captured.out

def test_main_invalid_source(tmp_path, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, 'argv', ["script.py", "--source", "/invalid/path"])
    migrate_ensemble_outputs.main()
    captured = capsys.readouterr()
    assert "跳过 (不存在)" in captured.out
