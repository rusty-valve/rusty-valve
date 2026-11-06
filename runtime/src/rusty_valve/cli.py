"""Portable local commands for the frozen Rusty Valve v0 materials."""
from pathlib import Path
import argparse
import importlib
import json
import os


def module(name, version='round2'):
    return importlib.import_module(f'rusty_valve.versions.{version}.valve_bench.{name}')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def score(record, output):
    from .public_exam_v1 import score_solved_log
    result = score_solved_log(record)
    save(output, result)
    print(json.dumps({'result': result['result'], 'output': str(output)}), flush=True)
    return result


def teacher(task, output):
    runtime = module('v2_episode')
    folder = Path(runtime.__file__).parent / 'resources/teacher' / task
    job = json.loads((folder / 'job.json').read_text())
    for key in ['storyboard_file', 'reviewed_natural_rest_file',
                'reviewed_free_connection_file', 'confirmed_sequence_reference_file']:
        job[key] = str((folder / job[key]).resolve())
    result = runtime.run_episode(job, output.resolve(), record=True)
    evidence = output / 'physical_step_evidence.jsonl.gz'
    if evidence.exists():
        score(evidence, output / 'public_score_v1.json')
    if result.get('error'):
        raise RuntimeError(result['error'])


def act(materials, version, index, output):
    materials = materials.resolve()
    data_name = 'four_tasks_646' if version == 'round1' else 'a2_close_coverage_300'
    os.environ['PRIMATES_VALVE_LEARNING_ROOT'] = str(materials / 'datasets' / data_name)
    os.environ['PRIMATES_VALVE_DATA_ROOT'] = str(materials)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(2)
    runtime = module('h3_act_eval', version)
    os.environ['PRIMATES_VALVE_ASSET_ROOT'] = str(Path(runtime.__file__).parent / 'assets')
    plan = json.loads((materials / 'evaluations' / version / 'examination_plan.json').read_text())
    if index < 0 or index >= len(plan):
        raise ValueError(f'case-index must be in [0, {len(plan) - 1}]')
    case = plan[index]
    result = runtime.run_exam(materials / 'models' / version / 'checkpoint',
        materials / case['source'], case['profile'], output.resolve(), None, device='cpu')
    score(output / 'physical_step_evidence.jsonl.gz', output / 'public_score_v1.json')
    save(output / 'invocation.json', {'round': version, 'case_index': index,
        'case': case, 'purpose': 'local installation check; does not replace published results'})
    if result.get('error'):
        raise RuntimeError(result['error'])


def data_check(materials, dataset, output):
    tasks_by_dataset = {
        'four_tasks_646': ('A1_close', 'A1_open', 'A2_close', 'A2_open'),
        'a2_close_coverage_300': ('A2_close',),
    }
    names = tasks_by_dataset if dataset == 'all' else (dataset,)
    expected = [
        (name, materials / 'datasets' / name / 'datasets' / task / split / 'meta' / 'info.json')
        for name in names
        for task in tasks_by_dataset[name]
        for split in ('train', 'validation')
    ]
    missing = [str(info) for _, info in expected if not info.is_file()]
    if missing:
        raise FileNotFoundError(
            'Missing expected LeRobot split metadata. Point --materials at the complete material bundle.\n'
            + '\n'.join(missing)
        )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rows = []
    for name, info in expected:
        root = info.parents[1]
        ds = LeRobotDataset(f'primates-valve/v0_{root.parent.name}_{root.name}',
            root=root.resolve(), video_backend='pyav', download_videos=False)
        row = ds[0]
        result = {'dataset': name, 'task': root.parent.name, 'split': root.name,
                  'episodes': ds.meta.total_episodes, 'frames': len(ds),
                  'action_shape': list(row['action'].shape),
                  'images': {k: list(v.shape) for k, v in row.items() if k.startswith('observation.images.')}}
        rows.append(result)
        print(json.dumps(result), flush=True)
        del ds
    if output:
        save(output, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('scene', help='Export a station XML and check that MuJoCo loads it.')
    p.add_argument('--station', choices=['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], required=True)
    p.add_argument('--output', type=Path, required=True, help='Path for the station XML checked by MuJoCo.')
    p = commands.add_parser('teacher', help='Run an OpenArm + S-hook script and write its Public Exam v1 score.')
    p.add_argument('--task', choices=['A1_open', 'A1_close', 'A2_open', 'A2_close'], required=True)
    p.add_argument('--output', type=Path, required=True, help='Directory for the scripted OpenArm + S-hook demonstration and its Public Exam v1 score.')
    p = commands.add_parser('act', help='Run a frozen ACT checkpoint on one fixed exam case using CPU inference.')
    p.add_argument('--materials', type=Path, required=True, help='Local material root containing the matching dataset, checkpoint, exam plan, and fixed inputs.')
    p.add_argument('--round', choices=['round1', 'round2'], required=True)
    p.add_argument('--case-index', type=int, default=0, help='Zero-based index in the selected exam plan.')
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('score', help='Apply Public Exam v1 to a complete physical-step evidence record.')
    p.add_argument('record', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('data-check', help='Read the first sample from every train and validation split in the selected datasets.')
    p.add_argument('--materials', type=Path, required=True, help='Local material root containing the complete selected LeRobot datasets.')
    p.add_argument('--dataset', choices=['all', 'four_tasks_646', 'a2_close_coverage_300'], default='all')
    p.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'scene':
        import mujoco
        xml = module('h0_scene').build_workstation(args.station)
        model = mujoco.MjModel.from_xml_string(xml)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(xml)
        print(json.dumps({'station': args.station, 'joints': model.njnt, 'actuators': model.nu, 'loaded': True}))
    elif args.command == 'teacher':
        teacher(args.task, args.output)
    elif args.command == 'act':
        act(args.materials, args.round, args.case_index, args.output)
    elif args.command == 'score':
        score(args.record, args.output)
    elif args.command == 'data-check':
        try:
            data_check(args.materials, args.dataset, args.output)
        except FileNotFoundError as error:
            parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
