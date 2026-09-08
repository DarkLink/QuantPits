"""Read-only legacy prediction-copy continuity, bound to sealed model hashes.

Live file-backend ancestry is an additional observed input, never substituted
for sealed model bytes. No MLflow client, model loading, or latest selection.
"""
from pathlib import Path
import re

from quantpits.training.model_identity import model_content_digest, trace_training_origin


def observe_model_copy_pair(reference_root, reference_manifest, current_root, current_manifest,
                            *, retained_guards=None, input_inventory=None):
    from quantpits.research import decision_surface as s
    guards, selected, inventories = [], [], []
    watched = set()
    namespaces = set()
    roots = (Path(reference_root), Path(current_root))

    def watch(root, path):
        if path in watched:
            return
        watched.add(path)
        relative = path.relative_to(root).as_posix()
        # The existing reader rejects symlinks, including intermediate parents.
        s._physical_path(path, "model input", directory=path.is_dir())
        guard = s.SourceMutationObserver(root, (relative,))
        guards.append(guard)
        if not guard.supported:
            raise s._ComponentIncomparable("MODEL_INPUT_OBSERVATION_UNSUPPORTED")
        before = s._selected_fingerprint((path,))
        selected.append((path, before))

    def read(root, path):
        watch(root, path)
        return s._read_regular(path, maximum=128 * 1024 * 1024)[0]

    def projection(root, manifest):
        source = s._source_projection(manifest)  # authoritative partition validation
        artifacts = {a['position']: a for a in manifest['model_and_ensemble_lineage']['source_artifacts']
                     if a.get('role') == 'source_training'}
        rows = []
        for member in source['members']:
            artifact = artifacts[member['position']]
            locator = Path(artifact['artifact_locator'])
            if (locator.is_absolute() or len(locator.parts) != 4 or locator.parts[0] != 'mlruns'
                    or locator.parts[-1] != 'artifacts' or locator.parts[2] != artifact['recorder_id']
                    or any(part in ('.', '..') for part in locator.parts)):
                raise s._ComponentIncomparable('MODEL_LINEAGE_BACKEND_UNSUPPORTED')
            experiment_directories = {}

            def tags(experiment, identifier):
                if not re.fullmatch(r'[a-zA-Z0-9_-]+', identifier):
                    raise ValueError('invalid recorder selector')
                if experiment not in experiment_directories:
                    # Resolve an explicit experiment name, not a latest recorder.
                    import yaml
                    mlruns = root / 'mlruns'
                    if mlruns not in namespaces:
                        s._physical_path(mlruns, "model backend", directory=True)
                        namespace_guard = s.SourceMutationObserver(mlruns, ())
                        guards.append(namespace_guard)
                        if not namespace_guard.supported:
                            raise s._ComponentIncomparable("MODEL_INPUT_OBSERVATION_UNSUPPORTED")
                        # Watch the experiment namespace shallowly; never inventory
                        # all recorder payloads just to resolve an experiment name.
                        namespace_guard._watch(mlruns, namespace_guard._SELF_MASK | namespace_guard._PARENT_MASK, None, True)
                        for directory in sorted(mlruns.iterdir()):
                            if directory.is_dir():
                                guard = s.SourceMutationObserver(root, ((directory / 'meta.yaml').relative_to(root).as_posix(),))
                                guards.append(guard)
                                if not guard.supported:
                                    raise s._ComponentIncomparable("MODEL_INPUT_OBSERVATION_UNSUPPORTED")
                        namespaces.add(mlruns)
                    inventory = tuple(sorted(mlruns.glob("*/meta.yaml")))
                    inventories.append((mlruns, inventory))
                    matches = []
                    for path in inventory:
                        value = yaml.safe_load(read(root, path))
                        if isinstance(value, dict) and value.get('name') == experiment:
                            matches.append(path.parent)
                    if len(matches) != 1:
                        raise s._ComponentIncomparable('MODEL_LINEAGE_EXPERIMENT_AMBIGUOUS')
                    experiment_directories[experiment] = matches[0]
                directory = experiment_directories[experiment] / identifier
                if identifier == artifact['recorder_id'] and directory != root / locator.parent:
                    raise ValueError('source experiment locator mismatch')
                if not directory.is_dir():
                    raise s._ComponentIncomparable('MODEL_LINEAGE_RECORD_MISSING')
                watch(root, directory / 'tags')
                values = {path.name: read(root, path).decode('utf-8')
                          for path in sorted((directory / 'tags').iterdir()) if path.is_file()}
                return values

            origin = trace_training_origin(artifact['experiment_name'], artifact['recorder_id'], tags,
                                           member['family'])
            names, auxiliary = {}, {}
            for item in artifact['members']:
                path = Path(item['path'])
                try:
                    relative = path.relative_to(locator).as_posix()
                except ValueError:
                    raise s._ComponentIncomparable('MODEL_MEMBER_LOCATOR_INVALID')
                if relative == 'model.pkl' or re.fullmatch(r'model_fold_[0-9]+\.pkl', relative):
                    raw = read(root, root / path)
                    if s._digest(raw, 'raw_bytes') != item['digest']:
                        raise s._ComponentIncomparable('MODEL_SEALED_BYTES_MISMATCH')
                    names[relative] = model_content_digest(raw)
                elif (relative not in {'pred.pkl', 'label.pkl', 'code_status.txt', 'code_diff.txt', 'code_cached.txt'}
                      and not relative.startswith(('portfolio_analysis/', 'sig_analysis/'))):
                    # Only explicitly known observation/report artifacts are excluded.
                    # Unknown inputs/configuration remain exact content dependencies.
                    raw = read(root, root / path)
                    if s._digest(raw, 'raw_bytes') != item['digest']:
                        raise s._ComponentIncomparable('MODEL_SEALED_BYTES_MISMATCH')
                    auxiliary[relative] = s._digest(raw, 'raw_bytes')
            if not names or ('model.pkl' in names and len(names) != 1):
                raise s._ComponentIncomparable('MODEL_MEMBER_SET_INVALID')
            if 'model.pkl' not in names and set(names) != {'model_fold_%d.pkl' % i for i in range(len(names))}:
                raise s._ComponentIncomparable('MODEL_FOLD_SET_INVALID')
            rows.append({'position': member['position'], 'source_id': member['source_id'],
                         'origin': origin, 'models': names, 'auxiliary_inputs': auxiliary})
        return {'protocol': 'OBSERVED_TRAINING_ORIGIN_MODEL_CONTENT_V1', 'members': rows}

    try:
        values = (projection(roots[0], reference_manifest), projection(roots[1], current_manifest))
        if (any(before != s._selected_fingerprint((path,)) for path, before in selected)
                or any(g.mutated() for g in guards)
                or any(tuple(sorted(root.glob('*/meta.yaml'))) != entries for root, entries in inventories)):
            raise s._ComponentIncomparable('MODEL_INPUT_CHANGED')
        if input_inventory is not None:
            input_inventory.append(s._digest([before for _, before in selected]))
        if retained_guards is not None:
            retained_guards.extend(guards)
            guards.clear()  # Ownership transfers only after successful verification.
        return values
    finally:
        for guard in reversed(guards):
            guard.close()
