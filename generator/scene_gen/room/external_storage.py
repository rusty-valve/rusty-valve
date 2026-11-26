"""Check whether the output location has enough disk space."""
import math
from pathlib import Path
import shutil

from .recipe import InputRejected


def require_external_output(output, required_gib=30., *, separate_drive=False):
    if not math.isfinite(required_gib) or required_gib <= 0:
        raise InputRejected('The disk budget must be positive')
    target = Path(output).expanduser().resolve()
    parent = target
    while not parent.exists():
        parent = parent.parent
    if separate_drive and parent.stat().st_dev == Path(parent.anchor).stat().st_dev:
        raise InputRejected('Choose an output directory on a different drive')
    if shutil.disk_usage(parent).free < required_gib * 1024**3:
        raise InputRejected('The output filesystem has insufficient free space')
    return {'status': 'ready', 'output': str(target), 'required_gib': required_gib}
