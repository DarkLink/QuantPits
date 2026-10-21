"""Native evidence and narrow adapters. No account state is written here."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta
from pathlib import Path

from quantpits.operations.config import CycleError, contained, path_at, read_json
from quantpits.research.paper_market import PaperMarket, load_signal
from quantpits.research.paper_workflow import PaperWorkflow
from quantpits.training.persistence import atomic_write_bytes, sha256_bytes
from quantpits.utils.workspace import fingerprint_file, fingerprint_value


def save_json(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)
    if path.read_bytes() != payload:
        raise CycleError("atomic write postcondition failed: %s" % path)


def fixed_bytes(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise CycleError("fixed cycle artifact changed: %s" % path)
    else:
        atomic_write_bytes(path, raw)
    if path.read_bytes() != raw:
        raise CycleError("fixed artifact write postcondition failed")


def ref(path):
    return {"path": str(path), "sha256": fingerprint_file(path)}


def next_day(value):
    return (datetime.strptime(value, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


def valuation_processed(result):
    """Accept known NAVs from an already validated native journal, including cached prices."""
    return result["status"] in ("COMPLETE", "STALE") and result["nav"] is not None


def _valuation_at(journal, selected):
    values = [v["result"] for v in journal["valuations"] if v["result"]["date"] == selected]
    if values:
        return values[-1]
    if journal["bootstrap"]["date"] == selected:
        return journal["bootstrap"]["valuation"]
    return None


def native_manifest(ctx, command, run_id):
    path = contained(ctx.root, "output/manifests/%s/%s.json" % (command, run_id))
    if not path.is_file():
        return None
    value = read_json(path)
    if value.get("command") != command or value.get("run_id") != run_id:
        raise CycleError("native manifest identity mismatch")
    return (path, value) if value.get("status") == "success" else None


def verify_refs(evidence):
    for item in evidence.get("files", []):
        if fingerprint_file(item["path"]) != item["sha256"]:
            raise CycleError("completed artifact changed: %s" % item["path"])


def validate_write_context(ctx):
    from quantpits.runtime.mlflow_integrity import resolve_mlflow_resource_uri
    for relative in ("config", "data", "output", "archive", "data/locks", "data/operator_log.jsonl",
                     "data/run_state.json", "data/training_transactions", "data/training_runs", "data/training_attempts",
                     "data/.post_trade_transactions", "data/.post_trade.lock", "data/paper", "output/cycles",
                     "output/mlflow_artifacts", "output/manifests/static_train", "output/manifests/ensemble_fusion",
                     "output/manifests/order_gen", "output/manifests/post-trade"):
        contained(ctx.root, relative)
    backend = resolve_mlflow_resource_uri(ctx.mlflow_uri, workspace_root=ctx.root)
    if not backend.contained:
        raise CycleError("writer backend must stay inside its declared workspace")


class NativeSteps:
    def __init__(self, project, request, runner, directory):
        self.project, self.request, self.runner, self.directory = project, request, runner, directory

    def ctx(self, step):
        return self.project.members[step["member"]].ctx if "member" in step else None

    def signal_dir(self, step):
        return self.project.signal_dir(self.request["cycle"], step["signal"])

    def paper(self, step):
        return PaperWorkflow(self.ctx(step))

    def valuation(self, step):
        journal = self.paper(step)._journal()
        return _valuation_at(journal, step.get("date", self.request["value_through"]))

    def invoke(self, step, module, arguments):
        return self.runner.run(self.ctx(step), "quantpits.scripts." + module, arguments,
                               contained(self.directory, "logs/%s.log" % step["id"]))

    def observe(self, step, *, completed=False):
        """Return proof of completion, or None. Called before any repeated command."""
        kind = step["kind"]
        ctx = self.ctx(step)
        if kind == "predict":
            return self._prediction_snapshot(step, completed=completed)
        if kind == "fuse":
            return self._ensemble_snapshot(step, completed=completed)
        if kind in ("settle", "value", "paper_plan"):
            workflow = self.paper(step)
            journal = workflow._journal()  # validates the authoritative journal, read only
            if kind == "settle":
                result = workflow._settled(journal).get(step["native_cycle"])
                if result is None:
                    return None
            elif kind == "value":
                selected = step.get("date", self.request["value_through"])
                result = _valuation_at(journal, selected)
                if result is None or not valuation_processed(result):
                    return None
            else:
                result = journal["plans"].get(step["native_cycle"])
                if result is None:
                    return None
                if result["anchor"] != self.request["anchor"] or result["trade_date"] != self.request["trade_date"]:
                    raise CycleError("existing native paper plan has different dates")
                snapshot = self._signal_snapshot(step)
                if result["source"]["sha256"] != snapshot["predictions"][snapshot["default_combo"]]["sha256"]:
                    raise CycleError("existing paper plan uses a different signal")
            if kind in ("settle", "paper_plan"):
                rule = self.rules_path(step, freeze=False)
                if rule and read_json(rule) != result["rules" if kind == "settle" else "market_rules"]:
                    raise CycleError("saved paper result uses different explicitly requested rules")
            return {"native_cycle": step.get("native_cycle"), "journal": str(workflow.store.path),
                    "result_digest": fingerprint_value(result), "files": []}
        if kind in ("post_trade", "post_recover"):
            from quantpits.post_trade.transaction import PostTradeTransactionManager
            manager = PostTradeTransactionManager(ctx)
            if step.get("already_current"):
                cursor = read_json(ctx.config_path("prod_config.json"))
                if cursor.get("last_processed_date", cursor.get("current_date")) < step["through"] or manager.active():
                    raise CycleError("actual cursor is no longer at the saved frontier")
                return {"reason": "already at requested cursor", "files": []}
            path = manager.journal_path(step["run_id"])
            if not path.exists():
                return None
            journal = manager.load(step["run_id"])
            if journal.status != "completed":
                return None
            if journal.cursor_after != step["through"] or journal.scope != step["scope"]:
                raise CycleError("native actual transaction differs from cycle request")
            if not completed:
                verified = manager.verified_target_paths(journal)
                if set(verified) != {a.path for a in journal.artifacts}:
                    raise CycleError("actual transaction postimages differ from committed targets")
            manifest = native_manifest(ctx, "post-trade", step["run_id"])
            if manifest is None or manifest[1]["records"].get("transaction_id") != step["run_id"]:
                return None
            if manifest[1]["records"].get("output_fingerprints") != {a.path: a.target_sha256 for a in journal.artifacts}:
                raise CycleError("actual manifest output ledger differs from its native transaction")
            return {"manifest": str(manifest[0]), "run_id": step["run_id"], "files": [ref(manifest[0])]}
        if kind == "order":
            manifest = native_manifest(ctx, "order_gen", step["run_id"])
            if manifest is None:
                return None
            records = manifest[1]["records"]
            snapshot = self._signal_snapshot(step)
            if (records.get("anchor_date"), records.get("trade_date"), records.get("source", {}).get("record_id")) != (
                self.request["anchor"], self.request["trade_date"], snapshot["combos"][snapshot["default_combo"]],
            ):
                raise CycleError("order manifest dates/source differ from project request")
            outputs = [contained(ctx.root, item["path"]) for item in manifest[1]["outputs"]]
            if not outputs or any(not p.is_file() for p in outputs):
                raise CycleError("order manifest output is missing")
            return {"run_id": step["run_id"], "manifest": str(manifest[0]), "files": [ref(p) for p in outputs]}
        if kind == "rules":
            path = contained(self.directory, "rules.json")
            observation = contained(self.directory, "rules_observation.json")
            if not path.exists() or not observation.exists():
                return None
            from quantpits.research.paper_accounting import validate_rules
            from quantpits.research.paper_market import verify_closes
            rules, facts = read_json(path), read_json(observation)
            validate_rules(rules, self.request["trade_date"])
            if facts["anchor"] != self.request["anchor"]:
                raise CycleError("prepared reference observation has a different anchor")
            verify_closes(facts["receipts"], facts["anchor"], facts["closes"])
            return {"files": [ref(path), ref(observation)]}
        return None  # derived report is safe to regenerate after an interrupted writer

    def _prediction_snapshot(self, step, *, completed=False):
        from quantpits.training.records import TrainingRecordSnapshot, snapshot_from_dict
        from quantpits.training.record_repository import TrainingRecordRepository
        from quantpits.training.publication import TrainingPublicationIntent, TrainingPublicationReceipt, TrainingPublicationCoordinator
        directory, ctx = self.signal_dir(step), self.ctx(step)
        target = contained(ctx.root, directory / "predictions.json")
        proof = contained(ctx.root, directory / "prediction_receipt.json")
        if completed and (not target.exists() or not proof.exists()):
            raise CycleError("completed prediction has no fixed native evidence")
        if completed:
            saved = read_json(proof)
            if saved.get("run_id") != step["run_id"] or len(saved.get("files", [])) != 4:
                raise CycleError("invalid completed prediction proof")
            verify_refs(saved)
        manifest = native_manifest(ctx, "static_train", step["run_id"])
        if manifest is None:
            return None
        # An unfinished native closure must be resumed before adopting its outputs.
        state_path = ctx.data_path("run_state.json")
        if state_path.exists() and read_json(state_path).get("run_id") == step["run_id"]:
            return None
        records = manifest[1]["records"]
        keys = [m + "@static" for m in self.project.signals[step["signal"]]["models"]]
        outcomes = records.get("outcomes", [])
        if records.get("anchor_date") != self.request["anchor"] or set(records.get("target_keys", [])) != set(keys) or len(outcomes) != len(keys) or {o.get("key") for o in outcomes} != set(keys) or any(o.get("outcome") != "success" or o.get("published") is not True for o in outcomes):
            raise CycleError("native prediction did not complete the requested members/anchor")
        transaction = records.get("publication", {}).get("transaction_id")
        folder = contained(ctx.root, "data/training_transactions/" + str(transaction))
        intent_path, receipt_path = folder / "intent.json", folder / "receipt.json"
        intent = TrainingPublicationIntent.from_dict(read_json(intent_path))
        receipt = TrainingPublicationReceipt.from_dict(read_json(receipt_path))
        # Use the existing native validator, without requiring its mutable publication
        # still to be current. Staged postimages remain the original source.
        validator = object.__new__(TrainingPublicationCoordinator)
        validator.verify_receipt(intent, receipt, verify_current=False)
        if intent.run_id != step["run_id"] or set(receipt.published_keys) != set(keys):
            raise CycleError("prediction receipt belongs to another request")
        member = next(m for m in intent.members if m.kind == "record")
        raw = contained(folder, member.staged_relative_path).read_bytes()
        if sha256_bytes(raw) != member.postimage_fingerprint:
            raise CycleError("original prediction publication postimage is corrupt")
        snapshot = snapshot_from_dict(json.loads(raw.decode("utf-8")))
        entries = []
        observed = []
        fit = snapshot_from_dict(read_json(directory / "fit_records.json"))
        for key in keys:
            entry = snapshot.entry_map[key]
            outcome = next(o for o in outcomes if o["key"] == key)
            if entry.status != "ready" or entry.requested_anchor != self.request["anchor"] or entry.recorder_id != outcome["recorder_id"]:
                raise CycleError("prediction record does not match successful native result")
            source = fit.entry_map[key]
            if (entry.source_recorder_id, entry.source_experiment_name, entry.source_operation) != (source.recorder_id, source.experiment_name, source.operation):
                raise CycleError("prediction does not use the selected fixed fit identity")
            _, identity = load_signal(ctx, {"recorder_id": entry.recorder_id, "experiment_id": entry.experiment_id}, self.request["anchor"])
            from quantpits.research.replay import _prediction_frame
            raw_prediction = Path(identity["path"]).read_bytes()
            if sha256_bytes(raw_prediction) != identity["sha256"]:
                raise CycleError("prediction changed during snapshot observation")
            actual = _prediction_frame(raw_prediction, key)
            if (min(actual), max(actual), sum(len(rows) for rows in actual.values())) != (entry.prediction_start, entry.prediction_end, entry.prediction_rows):
                raise CycleError("prediction actual coverage differs from its native record")
            entries.append(entry)
            observed.append({"key": key, "recorder_id": entry.recorder_id, "sha256": identity["sha256"]})
        fixed_bytes(target, TrainingRecordRepository.serialize(TrainingRecordSnapshot(tuple(entries))))
        evidence = {"run_id": step["run_id"], "manifest": str(manifest[0]), "source": str(target),
                    "members": observed, "files": [ref(target), ref(manifest[0]), ref(intent_path), ref(receipt_path)]}
        if proof.exists():
            if read_json(proof) != evidence:
                raise CycleError("prediction proof differs from original native result")
        elif not completed:
            save_json(proof, evidence)
        return evidence

    def _ensemble_snapshot(self, step, *, completed=False):
        directory, ctx = self.signal_dir(step), self.ctx(step)
        target, proof = directory / "ensemble.json", directory / "ensemble_receipt.json"
        if completed and (not target.exists() or not proof.exists()):
            raise CycleError("completed fusion has no fixed native evidence")
        if completed:
            saved = read_json(proof)
            if saved.get("run_id") != step["run_id"] or len(saved.get("files", [])) != 3 or not (directory / "signal.csv").is_file():
                raise CycleError("invalid completed fusion proof")
            verify_refs(saved)
        manifest = native_manifest(ctx, "ensemble_fusion", step["run_id"])
        if manifest is None:
            return None
        spec = self.project.signals[step["signal"]]
        prediction = read_json(directory / "predictions.json")
        records = manifest[1]["records"]
        combos = records.get("combos", [])
        if records.get("anchor_date") != self.request["anchor"] or len(combos) != 1 or combos[0].get("name") != spec["combo"]:
            raise CycleError("ensemble manifest differs from requested combo/anchor")
        combo = combos[0]
        if set(combo.get("models", [])) != set(prediction["models"]) or len(combo["models"]) != len(prediction["models"]):
            raise CycleError("ensemble result has a different model set")
        inputs = records.get("input_models", [])
        if len(inputs) != len(prediction["models"]) or {i.get("resolved_key"): i.get("recorder_id") for i in inputs} != prediction["models"] or any(i.get("prediction_end") != self.request["anchor"] or i.get("status") != "ready" for i in inputs):
            raise CycleError("ensemble input evidence differs from this cycle predictions")
        output = combo.get("output_evidence", {})
        rid = combo.get("recorder_id")
        if not rid or output.get("recorder_id") != rid or not output.get("experiment_id"):
            raise CycleError("ensemble output has no verified recorder identity")
        scores, identity = load_signal(ctx, {"recorder_id": rid, "experiment_id": output["experiment_id"]}, self.request["anchor"])
        csv_path = directory / "signal.csv"
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["datetime", "instrument", "score"])
        for instrument, score in sorted(scores.items()):
            writer.writerow([self.request["anchor"], instrument, "" if score is None else score])
        fixed_bytes(csv_path, stream.getvalue().encode("utf-8"))
        snapshot = {"combos": {spec["combo"]: rid}, "default_combo": spec["combo"],
                    "anchor_date": self.request["anchor"], "run_id": step["run_id"],
                    "source_recorders": prediction["models"], "workspace": str(ctx.root), "mlflow_uri": ctx.mlflow_uri,
                    "combo_meta": {spec["combo"]: output},
                    "predictions": {spec["combo"]: {"path": str(csv_path), "sha256": fingerprint_file(csv_path)}},
                    "native_prediction": identity}
        fixed_bytes(target, (json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        evidence = {"run_id": step["run_id"], "manifest": str(manifest[0]), "source": str(target),
                    "files": [ref(target), ref(csv_path), ref(manifest[0])]}
        if proof.exists():
            if read_json(proof) != evidence:
                raise CycleError("fusion proof differs from original native result")
        elif not completed:
            save_json(proof, evidence)
        return evidence

    def _signal_snapshot(self, step):
        return read_json(self.signal_dir(step) / "ensemble.json")

    def execute(self, step):
        kind, request, ctx = step["kind"], self.request, self.ctx(step)
        if kind == "blocked":
            return 2
        if kind == "predict":
            from quantpits.training.command import TrainingRunOptions, prepare_training_run
            directory = self.signal_dir(step)
            directory.mkdir(parents=True, exist_ok=True)
            spec = self.project.signals[step["signal"]]
            source = contained(ctx.root, spec["fit_records"])
            frozen = directory / "fit_records.json"
            if not frozen.exists():
                raw = source.read_bytes()
                if sha256_bytes(raw) != step["fit_sha256"]:
                    raise CycleError("selected fit source changed since request capture")
                fixed_bytes(frozen, raw)
            state = ctx.data_path("run_state.json")
            resume = state.exists()
            if resume and read_json(state).get("run_id") != step["run_id"]:
                raise CycleError("producer has another unfinished native run; resolve it first")
            options = TrainingRunOptions(family="static", action="predict_only", models=tuple(spec["models"]),
                                         source_records=str(frozen), anchor=request["anchor"], run_id=step["run_id"], resume=resume)
            prepare_training_run(ctx=ctx, options=options)  # exact native config validation before launching
            arguments = ["--predict-only", "--models", ",".join(spec["models"]), "--source-records", str(frozen),
                         "--anchor", request["anchor"], "--run-id", step["run_id"]]
            if resume:
                arguments.append("--resume")
            return self.invoke(step, "static_train", arguments)
        if kind == "fuse":
            from quantpits.ensemble.types import EnsembleRunOptions
            from quantpits.ensemble.service import prepare_ensemble_run
            spec = self.project.signals[step["signal"]]
            directory = self.signal_dir(step)
            source = str(directory / "predictions.json")
            options = EnsembleRunOptions(combo=spec["combo"], record_file=source, norm_method=spec.get("norm_method", "rank"))
            prepared = prepare_ensemble_run(ctx=ctx, options=options, cli_args=())
            keys = set(read_json(source)["models"])
            if len(prepared.combos) != 1 or set(prepared.combos[0].models) != keys:
                raise CycleError("configured fusion combo differs from the explicit signal model set")
            return self.invoke(step, "ensemble_fusion", ["--combo", spec["combo"], "--record-file", source,
                "--norm-method", spec.get("norm_method", "rank"), "--no-backtest", "--no-charts", "--save-csv",
                "--output-dir", str(directory / "fusion"), "--prediction-dir", str(directory / "native_predictions"), "--run-id", step["run_id"]])
        if kind in ("settle", "value", "paper_plan"):
            arguments = ["--workspace", str(ctx.root), "--mlflow-uri", ctx.mlflow_uri, "--qlib-data-dir", str(ctx.qlib_data_dir)]
            if kind == "settle":
                arguments += ["settle", "--cycle", step["native_cycle"]]
            elif kind == "value":
                arguments += ["value", "--date", step.get("date", request["value_through"])]
            else:
                snapshot = self._signal_snapshot(step)
                arguments += ["plan", "--cycle", step["native_cycle"], "--anchor", request["anchor"], "--trade-date", request["trade_date"],
                              "--prediction", snapshot["predictions"][snapshot["default_combo"]]["path"],
                              "--signal-workspace", snapshot["workspace"], "--signal-mlflow-uri", snapshot["mlflow_uri"],
                              "--lineage-file", str(self.signal_dir(step) / "ensemble.json"),
                              "--lineage-file", str(self.signal_dir(step) / "predictions.json")]
            rule = self.rules_path(step)
            if rule and kind != "value":
                arguments += ["--rules", str(rule)]
            return self.invoke(step, "paper_trade", arguments)
        if kind in ("post_trade", "post_recover"):
            arguments = ["--scope", step["scope"], "--start-date", step["start"], "--end-date", step["through"], "--run-id", step["run_id"]]
            spec = request["inputs"].get("actual", {}).get(step["member"], {})
            if spec.get("settlement_bundle"):
                arguments += ["--settlement-bundle", spec["settlement_bundle"]]
            if spec.get("allow_missing_settlement"):
                arguments.append("--allow-missing-settlement")
            return self.invoke(step, "prod_post_trade", arguments)
        if kind == "order":
            spec = self.project.signals[step["signal"]]
            return self.invoke(step, "order_gen", ["--anchor", request["anchor"], "--trade-date", request["trade_date"],
                "--combo", spec["combo"], "--ensemble-record-file", str(self.signal_dir(step) / "ensemble.json"),
                "--record-file", str(self.signal_dir(step) / "predictions.json"),
                "--output-dir", "output/cycles/%s/%s/orders" % (self.project.path.stem, request["cycle"]), "--run-id", step["run_id"]])
        if kind == "report":
            return self.invoke(step, "compare_accounts", ["--group", str(self.project.group.path), "--since", request["since"],
                "--date", request["value_through"], "--output", str(self.directory / "reports")])
        if kind == "rules":
            return self._prepare_rules(step)
        raise CycleError("unknown cycle step")

    def rules_path(self, step, *, freeze=True):
        if step.get("rules_source"):
            source = Path(step["rules_source"])
            target = contained(self.directory, "inputs/rules-%s.json" % fingerprint_value(str(source))[:16])
            if target.exists():
                if step.get("rules_sha256") and fingerprint_file(target) != step["rules_sha256"]:
                    raise CycleError("fixed rules digest mismatch")
                return target
            if not freeze:
                return source
            raw = source.read_bytes()
            if step.get("rules_sha256") and sha256_bytes(raw) != step["rules_sha256"]:
                raise CycleError("requested dated rules changed before use")
            fixed_bytes(target, raw)
            return target
        binding = self.project.accounts.get(step.get("member"), {})
        if step["kind"] == "settle":
            value = binding.get("pending_rules", {}).get(step["native_cycle"])
        else:
            value = binding.get("rules", self.project.rules)
            if value is None and self.project.prepare_rules is not None:
                return self.directory / "rules.json"
        return path_at(self.project.path.parent, value) if value else None

    def _prepare_rules(self, step):
        from quantpits.research.paper_accounting import validate_rules
        template = read_json(path_at(self.project.path.parent, self.project.prepare_rules))
        if template.get("reference_assumption") != "anchor_close_for_normal_session":
            raise CycleError("prepared rules require an explicit normal-session anchor-close assumption")
        template.pop("reference_assumption")
        market = PaperMarket(self.ctx(step).qlib_data_dir)
        if market.check_date(self.request["anchor"], actual=True) != "VERIFIED":
            return 2
        instruments = set()
        account_gaps = []
        for member in self.project.group.members:
            config = read_json(member.ctx.config_path("model_config.json"))
            pool, _ = market.universe(config["market"], self.request["anchor"])
            instruments.update(pool)
            if member.kind == "paper":
                from quantpits.research.paper_store import latest_state
                try:
                    instruments.update(p["instrument"] for p in latest_state(PaperWorkflow(member.ctx)._journal())["positions"])
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    account_gaps.append({"member": member.id, "error": str(exc)})
            else:
                instruments.update(p["instrument"] for p in read_json(member.ctx.config_path("prod_config.json")).get("current_holding", []))
        closes = market.closes(self.request["anchor"], instruments)
        template["date"] = self.request["trade_date"]
        template.setdefault("instruments", {})
        for instrument, close in closes.items():
            if close is not None:
                template["instruments"].setdefault(instrument, {}).setdefault("reference_close", close)
        validate_rules(template, self.request["trade_date"])
        save_json(self.directory / "rules.json", template)
        save_json(self.directory / "rules_observation.json", {"anchor": self.request["anchor"], "receipts": market.receipts,
                  "closes": closes, "account_gaps": account_gaps})
        return 0
