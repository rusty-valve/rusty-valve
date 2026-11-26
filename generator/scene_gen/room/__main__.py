"""Build a room from a recipe or open a room in MuJoCo."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from copy import deepcopy
import json
from pathlib import Path
import traceback
from .evaluation_common import portable_generation, require_environment

def main():
    if sys.argv[1:2] == ['generate']:
        from .generate_room import main as generate
        return generate(sys.argv[2:])
    if sys.argv[1:2] == ['simulate']:
        from .run_room import main as simulate
        return simulate(sys.argv[2:], default_room=None)
    environment = require_environment()
    from .recipe import InputRejected, batch_recipes
    from .building_equipment_generation import generate as generate_layout
    from .process import RULES, write_json
    from .shell import resolve_shell
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('layout','piping','full','valves','prune-valves','insulation','site'),default='layout',
                        help='Choose the room generation stage')
    parser.add_argument('--saved-room',type=Path,help='Saved model directory for later stages')
    parser.add_argument('--bund-height-mm',type=float,help='Containment curb height for site generation, default 300 mm')
    parser.add_argument('--threaded-lever-sample', metavar='VALVE_ID',
                        help=argparse.SUPPRESS)
    parser.add_argument('--equipment-reference',type=Path,help='Keep saved equipment positions in full generation')
    parser.add_argument('--previous-model', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--recovery-reference', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--layout-source',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--piping-method',choices=('end-bundle',),default='end-bundle',help=argparse.SUPPRESS)
    parser.add_argument('--trial-port-straights', type=Path,
                        help=argparse.SUPPRESS)
    parser.add_argument('--required-gib',type=float,default=30.,help=argparse.SUPPRESS)
    parser.add_argument('--recipe', '--input', type=Path, help='Process recipe JSON without coordinates or orientation')
    parser.add_argument('--shell', type=Path, help='Shell JSON with bays, platforms and storey_height_m')
    parser.add_argument('--bays', type=int, default=2)
    parser.add_argument('--platforms', type=int, default=2)
    parser.add_argument('--storey-height', type=float, default=RULES['construction_choices']['storey_height_m'])
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True,help='Save the results in a new directory')
    parser.add_argument('--skip-images', action='store_true')
    args = parser.parse_args()
    from .external_storage import require_external_output
    try:storage=require_external_output(args.output,args.required_gib)
    except InputRejected as exc:parser.error(str(exc))
    if args.trial_port_straights or args.previous_model or args.recovery_reference:
        parser.error('This option is available in the development version of the generator')
    if args.equipment_reference and not (args.stage=='full' and args.piping_method=='end-bundle'):
        parser.error('--equipment-reference is only available in full end-bundle generation')
    if args.stage=='valves':
        if args.saved_room is None or any((args.recipe,args.shell,args.layout_source,args.threaded_lever_sample)):
            parser.error('The valves stage requires a saved piped room')
        from .valve_positioning import generate as generate_valves
        print(json.dumps(generate_valves(args.saved_room,args.output),ensure_ascii=False),flush=True)
        return 0
    if args.stage=='prune-valves':
        if args.saved_room is None or any((args.recipe, args.shell, args.layout_source, args.threaded_lever_sample)):
            parser.error('The prune-valves stage requires a saved room with placed valves')
        from .close_valve_pruning import generate as prune_valves
        report = prune_valves(args.saved_room, args.output)
        print(json.dumps({'after':report['after'], 'discarded':len(report['removed'])}, ensure_ascii=False), flush=True)
        return 0
    if args.stage=='insulation':
        if args.saved_room is None or any((args.recipe, args.shell, args.layout_source)):
            parser.error('The insulation stage requires --saved-room')
        from .insulation import generate as generate_insulation
        report = generate_insulation(args.saved_room,args.output,threaded_lever_sample=args.threaded_lever_sample)
        print(json.dumps({'before_insulation':report['before_insulation'],'stage4':report['stage4'],
                          'insulated_pieces':len(report['pieces'])},ensure_ascii=False))
        return 0
    if args.bund_height_mm is not None and args.stage!='site':
        parser.error('--bund-height-mm is only available in the site stage')
    if args.stage=='site':
        if args.saved_room is None or any((args.recipe,args.shell,args.layout_source,args.threaded_lever_sample)):
            parser.error('The site stage requires a saved insulated room')
        from .site import generate as generate_site
        from .site_ground import BUND_HEIGHT
        height = BUND_HEIGHT if args.bund_height_mm is None else args.bund_height_mm/1000
        if not 0 < height < float('inf'):
            parser.error('Containment curb height must be positive')
        report = generate_site(args.saved_room, args.output, bund_height=height)
        print(json.dumps({'hard_violations': len(report['hard_violations']),
                          'floors': len(report['floors'])}, ensure_ascii=False), flush=True)
        return int(bool(report['hard_violations']))
    if args.saved_room or args.threaded_lever_sample:
        parser.error('--saved-room is for later stages and --threaded-lever-sample requires insulation')
    if args.stage=='piping':
        if args.layout_source is None:parser.error('The piping stage requires --layout-source')
        if args.recipe is not None or args.shell is not None:parser.error('The piping stage does not accept recipe or shell overrides')
        from .piping_method import generate_platform
        result=generate_platform(args.layout_source,args.output,storage=storage)
        print(json.dumps({'status':result['status'],'connections':{k:result['connections'][k] for k in ('expected','connected')},
            'valves':{k:result['manual_valves'][k] for k in ('expected','installed','actual_attached')},
            'penetrations':result['actual_material_penetration_count'],
            'orthogonal_checked':result['directions']['checked'],'elapsed_seconds':result['elapsed_seconds']},ensure_ascii=False),flush=True)
        return 0
    generate=generate_layout
    version='building_layout_v2_4'
    if args.stage=='full':
        if args.recipe is None:parser.error('Full generation requires --recipe')
        if args.layout_source is not None:parser.error('Full generation uses a recipe instead of --layout-source')
        from .material_room_generation import generate as generate_material_room
        def generate(recipe, shell, seed, folder, images):
            return generate_material_room(recipe, shell, seed, folder, images, storage=storage,
                equipment_reference=args.equipment_reference)
        version='material_room_generation_v1'
    if args.count < 1:
        parser.error('--count must be a positive integer')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Choose a new empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    shell = json.loads(args.shell.read_text()) if args.shell else {
        'bays': args.bays, 'platforms': args.platforms, 'storey_height_m': args.storey_height}
    try:
        resolve_shell(shell)
        if args.recipe:
            recipe = json.loads(args.recipe.read_text())
            identifier = recipe.get('id', args.recipe.stem)
            if not isinstance(identifier, str) or not identifier or Path(identifier).name != identifier or identifier in ('.', '..'):
                raise InputRejected('The recipe ID must be a single directory name')
            work = [(recipe, args.seed, identifier)]
        else:
            work = [(recipe, seed, f'room_{i+1:02}')
                    for i, (recipe, seed) in enumerate(batch_recipes(args.seed, args.count))]
    except (InputRejected, ValueError) as exc:
        write_json(output / 'input_rejected.json', {'generation': 'rejected', 'reason': str(exc), 'shell': shell})
        write_json(output / 'summary.json', {'version': version, 'stage':args.stage,'seed': args.seed, 'environment': environment,
                   'rooms': [{'room': 'input', 'seed': args.seed, 'generation': 'rejected', 'reason': str(exc)}]})
        print(json.dumps({'generation': 'rejected', 'reason': str(exc)}, ensure_ascii=False))
        return 0
    summary = {'version': version, 'stage':args.stage,'seed': args.seed, 'environment': environment,
               'shell': shell, 'rooms': [], 'batch_includes_reference': False}
    write_json(output / 'rules_snapshot.json', {'version': RULES['version']})
    for recipe, seed, identifier in work:
        request = {'recipe': deepcopy(recipe), 'shell': deepcopy(shell), 'seed': seed}
        try:
            row = generate(recipe, shell, seed, output / identifier, not args.skip_images)
            if row['generation'] == 'rejected':
                write_json(output / f'{identifier}_rejected.json', {'input': request, **row})
        except InputRejected as exc:
            row = {'room': identifier, 'seed': seed, 'generation': 'rejected', 'reason': str(exc)}
            write_json(output / f'{identifier}_rejected.json', {'input': request, **row})
        except Exception as exc:
            row = {'room': identifier, 'seed': seed, 'generation': 'fail', 'error': str(exc)}
            progress=output/identifier/'progress.json'
            if progress.exists():row['failed_stage']=json.loads(progress.read_text()).get('stage')
            write_json(output / f'{identifier}_failure.json', {'input': request, **row, 'traceback': traceback.format_exc()})
        summary['rooms'].append(row)
        write_json(output / 'summary.json', summary)
        print(json.dumps(row, ensure_ascii=False, default=lambda value:value.item() if hasattr(value,'item') else list(value)), flush=True)
    return 0

if __name__ == '__main__':
    with portable_generation():
        raise SystemExit(main())
