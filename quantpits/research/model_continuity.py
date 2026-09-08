"""Read-only legacy prediction-copy continuity, bound to sealed model hashes.

Live file-backend ancestry is an additional observed input, never substituted
for sealed model bytes. No MLflow client, model loading, or latest selection.
"""
from pathlib import Path
import os
import re

from quantpits.training.model_identity import model_content_digest, trace_training_origin

MAX_INPUT_BYTES = 128 * 1024 * 1024


def observe_model_copy_pair(reference_root, reference_manifest, current_root, current_manifest,
                            *, retained_guards=None, input_inventory=None):
    from quantpits.research import decision_surface as s
    guards, selected, inventories = [], [], []
    watched = set()
    content = {}
    semantic = {}
    namespaces = set()
    roots = (Path(reference_root), Path(current_root))

    # Physical continuity is local to this invocation, never semantic identity.
    def metadata(path):
        info = os.lstat(str(path))
        return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
                info.st_size, info.st_mtime_ns, info.st_ctime_ns)

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
        before = metadata(path)
        selected.append((path, before))

    def read(root, path, role):
        watch(root, path)
        data = s._read_regular(path, maximum=MAX_INPUT_BYTES)[0]
        digest = s._digest(data, 'raw_bytes')
        if content.setdefault(path, digest) != digest:
            raise s._ComponentIncomparable('MODEL_INPUT_CHANGED')
        semantic[(role, path.relative_to(root).as_posix())] = {
            'state': 'file', 'raw_digest': digest,
        }
        return data

    def projection(root, manifest, role):
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
            resolved_locations = {}

            def tags(experiment, identifier):
                directory = resolved_locations[(experiment, identifier)]
                if identifier == artifact['recorder_id'] and directory != root / locator.parent:
                    raise ValueError('source experiment locator mismatch')
                if not directory.is_dir():
                    raise s._ComponentIncomparable('MODEL_LINEAGE_RECORD_MISSING')
                watch(root, directory / 'tags')
                values = {path.name: read(root, path, role).decode('utf-8')
                          for path in sorted((directory / 'tags').iterdir()) if path.is_file()}
                return values

            def resolve_identity(declared_experiment, identifier):
                # Run IDs, unlike legacy experiment-name tags, are stable. Resolve
                # exactly this ID across the explicit backend, never a latest run.
                if not isinstance(identifier, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', identifier):
                    raise s._ComponentIncomparable('MODEL_RECORDER_ID_INVALID')
                import yaml
                mlruns = root / 'mlruns'
                if mlruns not in namespaces:
                    s._physical_path(mlruns, "model backend", directory=True)
                    namespace_guard = s.SourceMutationObserver(mlruns, tuple(
                        directory.name + '/meta.yaml' for directory in mlruns.iterdir() if directory.is_dir()))
                    guards.append(namespace_guard)
                    if not namespace_guard.supported:
                        raise s._ComponentIncomparable("MODEL_INPUT_OBSERVATION_UNSUPPORTED")
                    namespace_guard._watch(mlruns, namespace_guard._SELF_MASK | namespace_guard._PARENT_MASK, None, True)
                    namespaces.add(mlruns)
                inventory = tuple(sorted(mlruns.glob('*/meta.yaml')))
                semantic[(role, 'mlruns')] = {
                    'state': 'experiment_inventory',
                    'members': [path.relative_to(mlruns).as_posix() for path in inventory],
                }
                inventories.append((mlruns, inventory))
                candidates = []
                for experiment_meta in inventory:
                    experiment_data = yaml.safe_load(read(root, experiment_meta, role))
                    record = experiment_meta.parent / identifier
                    guard = s.SourceMutationObserver(root, ((record / 'meta.yaml').relative_to(root).as_posix(),))
                    guards.append(guard)
                    if not guard.supported:
                        raise s._ComponentIncomparable('MODEL_INPUT_OBSERVATION_UNSUPPORTED')
                    if not record.exists():
                        semantic[(role, record.relative_to(root).as_posix())] = {'state': 'absent'}
                    else:
                        metadata = yaml.safe_load(read(root, record / 'meta.yaml', role))
                        ids = [metadata.get(key) for key in ('run_id', 'run_uuid') if key in metadata]
                        if (not ids or any(value != identifier for value in ids)
                                or str(metadata.get('experiment_id')) != experiment_meta.parent.name
                                or not isinstance(experiment_data.get('name'), str)):
                            raise s._ComponentIncomparable('MODEL_RECORDER_METADATA_MISMATCH')
                        candidates.append((experiment_data['name'], identifier))
                        resolved_locations[(experiment_data['name'], identifier)] = record
                if len(candidates) != 1:
                    raise s._ComponentIncomparable('MODEL_RECORDER_ID_MISSING_OR_AMBIGUOUS')
                return candidates[0]

            origin = trace_training_origin(artifact['experiment_name'], artifact['recorder_id'], tags,
                                           member['family'], resolve_identity=resolve_identity)
            names, auxiliary = {}, {}
            for item in artifact['members']:
                path = Path(item['path'])
                try:
                    relative = path.relative_to(locator).as_posix()
                except ValueError:
                    raise s._ComponentIncomparable('MODEL_MEMBER_LOCATOR_INVALID')
                if relative == 'model.pkl' or re.fullmatch(r'model_fold_[0-9]+\.pkl', relative):
                    raw = read(root, root / path, role)
                    if s._digest(raw, 'raw_bytes') != item['digest']:
                        raise s._ComponentIncomparable('MODEL_SEALED_BYTES_MISMATCH')
                    names[relative] = model_content_digest(raw)
                elif (relative not in {'pred.pkl', 'label.pkl', 'code_status.txt', 'code_diff.txt', 'code_cached.txt'}
                      and not relative.startswith(('portfolio_analysis/', 'sig_analysis/'))):
                    # Only explicitly known observation/report artifacts are excluded.
                    # Unknown inputs/configuration remain exact content dependencies.
                    raw = read(root, root / path, role)
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
        values = (projection(roots[0], reference_manifest, 'reference'),
                  projection(roots[1], current_manifest, 'current'))
        if (any(before != metadata(path) for path, before in selected)
                or any(digest != s._digest(s._read_regular(path, maximum=MAX_INPUT_BYTES)[0], 'raw_bytes')
                       for path, digest in content.items())
                or any(g.mutated() for g in guards)
                or any(tuple(sorted(root.glob('*/meta.yaml'))) != entries for root, entries in inventories)):
            raise s._ComponentIncomparable('MODEL_INPUT_CHANGED')
        if input_inventory is not None:
            input_inventory.append(s._digest({
                'protocol': 'MODEL_COPY_INPUT_CONTENT_INVENTORY_V1',
                'members': [dict(value, role=role, relative_path=relative)
                            for (role, relative), value in sorted(semantic.items())],
            }))
        if retained_guards is not None:
            retained_guards.extend(guards)
            guards.clear()  # Ownership transfers only after successful verification.
        return values
    finally:
        for guard in reversed(guards):
            guard.close()
