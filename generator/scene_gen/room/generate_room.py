"""Build a complete MuJoCo room from a recipe."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from . import batch_memory
from .evaluation_common import portable_generation, require_environment


STAGES = ('full', 'valves', 'prune-valves', 'insulation', 'site', 'package')
VERSION = {'tag': 'v0.2', 'commit': '942ef72394f5260b14f796edba0aacbd4df299c3'}
ROOM_MEMORY = 3_000_000_000
ROOM_TIMEOUT = 600


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_plan(args, output):
    from .shell import resolve_shell

    inputs = read(args.recipe)
    if not isinstance(inputs, dict):
        raise ValueError('The recipe must be a JSON object')
    envelope = inputs if 'recipe' in inputs else {}
    recipe = envelope['recipe'] if envelope else inputs
    if not isinstance(recipe, dict):
        raise ValueError('The process recipe must be a JSON object')
    name = recipe.get('id')
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('The recipe ID must be a single directory name')
    shell = read(args.shell) if args.shell else envelope.get('shell', {
        'bays': 2, 'platforms': 2, 'storey_height_m': 3.6,
    })
    resolve_shell(shell)
    seed = args.seed if args.seed is not None else envelope.get('seed')
    if type(seed) is not int:
        raise ValueError('Supply an integer --seed or a room recipe with a seed')
    site = envelope.get('site', {})
    if not isinstance(site, dict):
        raise ValueError('Site settings must be a JSON object')
    height = site.get('bund_height_mm', 300)
    if type(height) not in (int, float) or not math.isfinite(height) or height <= 0:
        raise ValueError('The containment curb height must be positive')
    salt = site.get('pattern_seed_salt')
    if salt is not None and (not isinstance(salt, list) or len(salt) > 128
                             or any(type(byte) is not int or not 0 <= byte <= 255 for byte in salt)):
        raise ValueError('The site pattern seed salt must be a list of byte values')
    reference = args.equipment_reference
    if reference is None and (relative := envelope.get('equipment_reference')):
        if not isinstance(relative, str):
            raise ValueError('The equipment reference must name a directory')
        parent = args.recipe.resolve().parent
        reference = (parent / relative).resolve()
        if Path(relative).is_absolute() or not reference.is_relative_to(parent):
            raise ValueError('The equipment reference must stay inside the recipe directory')
    if reference is not None:
        reference = reference.resolve()
        if not all((reference / file).is_file() for file in ('layout.json', 'shell.json')):
            raise ValueError(f'The equipment reference is missing layout.json or shell.json: {reference}')
    from .external_storage import require_external_output
    storage = require_external_output(output, args.required_gib, separate_drive=args.require_external)
    return {'recipe': recipe, 'shell': shell, 'seed': seed, 'bund_height_mm': height,
            'reference': str(reference) if reference else None, 'storage': storage, 'pattern_seed_salt': salt}


def generate_stage(stage, work):
    plan = read(work / 'plan.json')
    reference = Path(plan['reference']) if plan['reference'] else None
    stage_output = work / stage
    if stage == 'full':
        from .material_room_generation import generate
        result = generate(plan['recipe'], plan['shell'], plan['seed'], stage_output,
                          storage=plan['storage'], equipment_reference=reference)
        required = ('room.xml', 'geometry.json', 'process.json', 'piping.json', 'measure.json')
        if result.get('substatus') == 'stage_exception' or not all((stage_output / 'model' / file).is_file()
                                                      for file in required):
            raise ValueError(f'Pipe routing did not complete. See {stage_output / "result.json"}')
        return
    source = work / STAGES[STAGES.index(stage) - 1] / 'model'
    if stage == 'valves':
        from .valve_positioning import generate
        generate(source, stage_output)
    elif stage == 'prune-valves':
        from .close_valve_pruning import generate
        generate(source, stage_output)
    elif stage == 'insulation':
        from .insulation import generate
        generate(source, stage_output)
    elif stage == 'site':
        from .site import generate
        report = generate(source, stage_output, bund_height=plan['bund_height_mm'] / 1000,
                          pattern_seed_salt=plan['pattern_seed_salt'])
        if report['hard_violations']:
            raise ValueError(f'Site checks failed. See {stage_output / "model/site.json"}')
    elif stage == 'package':
        from .package_room import apply_public_valve_names, export_room
        result = export_room({'folder': str(source.parent), 'id': plan['recipe']['id']},
                             stage_output, scene_version=VERSION, equipment_reference=reference)
        recipe_path = stage_output / 'recipe.json'
        recipe = read(recipe_path)
        if plan['pattern_seed_salt'] is not None:
            recipe['site']['pattern_seed_salt'] = plan['pattern_seed_salt']
        write(recipe_path, recipe)
        apply_public_valve_names(stage_output, plan['recipe']['id'])
        write(work / 'package.json', result)


def run_stage(stage, work, deadline):
    log = work / f'{stage}.log'
    peak = 0
    command = [sys.executable, '-B', '-m', 'scene_gen.room.generate_room', stage, str(work)]
    with log.open('w', encoding='utf-8') as stream, subprocess.Popen(
        command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
    ) as process:
        try:
            while process.poll() is None:
                group = [batch_memory.process_memory(pid, rss)
                         for pid, pgid, rss, _ in batch_memory.scan_processes() if pgid == process.pid]
                peak = max(peak, sum(current for current, _ in group),
                           max((maximum for _, maximum in group), default=0))
                if peak > ROOM_MEMORY:
                    raise RuntimeError(f'{stage} exceeded the 3 GB memory limit. See {log}')
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'Room generation exceeded 600 seconds. See {log}')
                time.sleep(batch_memory.POLL_SECONDS)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if process.returncode:
            raise RuntimeError(f'{stage} failed. See {log}')
    return peak


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m scene_gen.room generate', description=__doc__)
    parser.add_argument('--recipe', required=True, type=Path, help='A process recipe or a packaged room recipe.json')
    parser.add_argument('--seed', type=int, help='Override the seed saved in a packaged recipe')
    parser.add_argument('--shell', type=Path, help='Override the building shell in the recipe')
    parser.add_argument('--equipment-reference', type=Path, help='Reuse equipment positions from this directory')
    parser.add_argument('--out', type=Path, help='Save the room in a new directory')
    parser.add_argument('--output', dest='out', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--require-external', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--required-gib', type=float, default=3., help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.out is None:
        parser.error('Provide --out')
    output = args.out.absolute()
    if output.exists() or output.is_symlink():
        parser.error('The output path already exists. Choose a new directory')
    output = output.resolve()
    from .recipe import InputRejected
    try:
        with portable_generation():
            require_environment()
            plan = load_plan(args, output)
    except (InputRejected, OSError, ValueError) as error:
        parser.error(str(error))
    lock = Path(tempfile.gettempdir()) / f'primates-robot-scene-batch-{os.getuid()}.lock'
    with lock.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('A scene generation job is already running')
        return generate_room(output, plan, parser)


def generate_room(output, plan, parser):
    state = batch_memory.system_memory()
    if not state['pressure_normal'] or state['available_bytes'] < batch_memory.AVAILABLE_BYTES:
        parser.error('Insufficient free memory to start room generation')
    if batch_memory.other_scene_processes(batch_memory.scan_processes(), ()):
        parser.error('A scene generation or viewer process is already running')
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'.{output.name}-generation-', dir=output.parent))
    write(work / 'plan.json', plan)
    print(f'Run files: {work}', flush=True)
    deadline = time.monotonic() + ROOM_TIMEOUT
    records = {}
    try:
        for stage in STAGES:
            print(f'Starting {stage}', flush=True)
            records[stage] = run_stage(stage, work, deadline)
            write(work / 'memory.json', records)
        if output.exists() or output.is_symlink():
            raise RuntimeError('The output path appeared during generation. The candidate is in the run directory')
        package = read(work / 'package.json')
        (work / 'package').rename(output)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({'room': str(output / 'room.xml'), 'connections': package['connections'],
                      'installed_valves': package['installed_valves'],
                      'insulation_pieces': package['insulation_pieces'],
                      'peak_memory_bytes': max(records.values())}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    with portable_generation():
        require_environment()
        generate_stage(sys.argv[1], Path(sys.argv[2]))
