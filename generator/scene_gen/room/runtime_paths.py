"""Find the equipment and valve files used by the generator."""
import os
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1]
_BUNDLED = _PACKAGE / 'runtime'
RUNTIME = Path(os.environ['SCENE_GEN_RUNTIME']).expanduser().resolve() if 'SCENE_GEN_RUNTIME' in os.environ else (
    _BUNDLED if _BUNDLED.is_dir() else _PACKAGE.with_name('generator_runtime'))
DATA = Path(__file__).resolve().parent / 'data'
