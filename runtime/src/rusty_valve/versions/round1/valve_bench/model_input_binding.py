"""Keep reviewed joints, tool installation and execution inputs in one model.

Joint names alone cannot detect a fixed S-tool installation change.  This is
an input compatibility check; it introduces no task success or safety score.
"""
from __future__ import annotations


def validate_model_input_binding(job, storyboard, adapter, rest, connections):
    documents = dict(job=job, storyboard=storyboard, adapter=adapter,
                     rest=rest, connections=connections or {})
    requested = bool(job.get('robot_adapter_file')) or any(
        row.get('model_revision') for row in documents.values())
    if not requested:
        return None
    revision, adapter_id = adapter.get('model_revision'), adapter.get('adapter_id')
    if not isinstance(revision, str) or not revision or not adapter_id:
        raise ValueError('Versioned execution needs an explicit robot installation revision')
    for field in ('robot_adapter_file', 'reviewed_natural_rest_file',
                  'reviewed_free_connection_file'):
        if not job.get(field):
            raise ValueError(f'Versioned execution requires explicit {field}')
    for name, row in documents.items():
        actual_id = row.get('adapter_id') if name == 'adapter' else row.get('robot_adapter_id')
        if row.get('model_revision') != revision or actual_id != adapter_id:
            raise ValueError(f'{name} does not match the reviewed robot installation revision')
    if storyboard.get('execution_approved') is not True:
        raise ValueError('Changed installation needs its reviewed G1 before execution')
    declared_rest = storyboard.get('shared_natural_rest_file')
    if declared_rest:
        from pathlib import Path
        declared_path = Path(declared_rest)
        if not declared_path.is_absolute():
            declared_path = Path(job['storyboard_file']).parent / declared_path
        if declared_path.resolve() != Path(job['reviewed_natural_rest_file']).resolve():
            raise ValueError('G1 and job name different reviewed natural-rest files')
    return dict(model_revision=revision, robot_adapter_id=adapter_id,
                storyboard_matches_installation=True,
                natural_rest_matches_installation=True,
                free_connections_match_installation=True)
