"""Staged V2 execution with separately selected and reviewed change categories.

Step 4 uses measured G1 phase transitions; step 5 adds finite preferred-first
contact-radius choices. Earlier entries preserve their prior behavior.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from .g2_episode import DEFAULTS, G2Controller, _run_prepared_episode
from .slow_open_rule import load_slow_open_rule
from .storyboard_inputs import load_storyboard_inputs
from .tool_orientation_context import POLICY_VERSION, REST_FILE

ENTRY_VERSION = 'v2_step1_formal_inputs_v1'
REVIEW_SCOPE = 'v2_step1_preserved_pose_regression'


def require_orientation_review(approval):
    """Code-change authorization does not approve the new geometric motion."""
    confirmation = approval.get('stage_aware_orientation_confirmation', {})
    if (confirmation.get('policy_version') != POLICY_VERSION
            or confirmation.get('user_confirmed') is not True
            or not confirmation.get('user_text')):
        raise ValueError('Revised rest/connections and working-side examples need '
            'explicit user G1 confirmation before physical execution')


def run_episode(job, output, record=True):
    start = time.monotonic()
    procedure_version = job.get('procedure_version')
    if procedure_version not in ('A1-v2.4', 'A1-v2.5'):
        raise ValueError('V2 runs require an explicitly selected v2.4/v2.5 scoring policy')
    review = job.get('execution_review', {})
    step = int(job.get('v2_step', 1))
    scopes = {1: REVIEW_SCOPE, 2: 'v2_step2_candidate_A_regression',
              3: 'v2_step3_obstacle_regression',4:'v2_step4_measured_phase_regression',
              5:'v2_step5_contact_range_regression'}
    mount_revision = job.get('execution_kind') == 'reviewed_mount_revision'
    expected_scope = ('v2_reviewed_mount_and_free_connections_regression'
                      if mount_revision else scopes.get(step))
    if step not in scopes or review.get('scope') != expected_scope or not review.get('user_text'):
        raise ValueError('An explicit review for the selected V2 step is required')
    inputs = load_storyboard_inputs(job['storyboard_file'],
        expected_station=job.get('station'), expected_operation=job.get('variant'))
    if inputs.operation not in ('open', 'close'):
        raise ValueError('Current v0 scope contains ordinary open and close only')
    from .h0_robot import load_robot_adapter
    from .model_input_binding import validate_model_input_binding
    selected_adapter = load_robot_adapter(job.get('robot_adapter_file', 'openarm_s_hook'))
    rest_file = Path(job.get('reviewed_natural_rest_file', REST_FILE)).resolve()
    rest_document = json.loads(rest_file.read_text(encoding='utf-8'))
    connection_file = job.get('reviewed_free_connection_file')
    connection_document = (json.loads(Path(connection_file).read_text(encoding='utf-8'))
                           if connection_file else None)
    model_binding = validate_model_input_binding(job, inputs.storyboard,
        selected_adapter.to_dict(), rest_document, connection_document)
    if mount_revision and model_binding is None:
        raise ValueError('Mount-revision execution must declare its reviewed model binding')
    parameters = copy.deepcopy(job.get('parameters', {}))
    missing = sorted(set(DEFAULTS)-set(parameters))
    if missing:
        raise ValueError(f'Complete motion parameters required; missing {missing}')
    if parameters['r_m'] is None or parameters['force_N'] is None:
        raise ValueError('Baseline contact radius and force must be explicitly resolved')
    rule = load_slow_open_rule()
    if parameters.get('pause_hold_s', rule['pause_s']) != rule['pause_s']:
        raise ValueError('Inactive pause metadata must still match the current rule')
    parameters['pause_hold_s'] = rule['pause_s']
    parameters['slow_open_rule'] = rule
    parameters['seed'] = int(job['seed'])
    if job.get('release_reference_policy'):
        from .g1_release_reference import VERSION as RELEASE_VERSION
        if job['release_reference_policy'] != RELEASE_VERSION or step < 4:
            raise ValueError('G1 release reference requires the measured-state controller')
        parameters['release_reference_policy'] = RELEASE_VERSION
    if step >= 2:
        approval = inputs.storyboard.get('approval', {})
        if mount_revision:
            if (approval.get('user_confirmed') is not True
                    or approval.get('model_revision') != model_binding['model_revision']
                    or approval.get('robot_adapter_id') != model_binding['robot_adapter_id']
                    or not approval.get('user_original_text')):
                raise ValueError('Changed tool installation and G1 need explicit matching user confirmation')
        else:
            require_orientation_review(approval)
        parameters['tool_upright_required'] = True
        parameters['tool_body_names'] = copy.deepcopy(job['tool_body_names'])
        parameters['tool_orientation_policy'] = POLICY_VERSION
        parameters['reviewed_natural_rest_file'] = str(rest_file)
        parameters['upright_connection_policy'] = 'natural_to_preparation_continuous_v1'
        if connection_file:
            parameters['reviewed_free_connection_file'] = str(Path(connection_file).resolve())
            parameters['upright_connection_policy'] = 'approved_joint_endpoint_quintic_v1'
    if step >= 3:
        parameters['path_obstacles_required'] = True
        reference_routes = job.get('route_names_to_preserve', {})
        if (set(reference_routes) != {'reach_path_ranking', 'return_path_ranking'}
                or any(not isinstance(v, str) or not v for v in reference_routes.values())):
            raise ValueError('Step 3 requires the previous successful route names; no silent side change')
        parameters['route_names_to_preserve'] = copy.deepcopy(reference_routes)
    # Preserve all effective baseline values and their controller revision.
    # The new reader's version is separate from the reused controller version.
    for field in ('controller_version', 'feedback_version', 'lift_path',
                  'free_path_version', 'execution_revision'):
        if not parameters.get(field):
            raise ValueError(f'Baseline metadata missing {field}')
    offset = np.asarray(job.get('offset_m', [0., 0., 0.]), dtype=float)
    scale = float(job.get('resistance_scale', 1.))
    if offset.shape != (3,) or not np.isfinite(offset).all() or not .5 <= scale <= 1.5:
        raise ValueError('Need finite valve offset and declared resistance in [0.5, 1.5]')
    station = inputs.station
    approved_station = dict(placement=inputs.placement, cameras=inputs.cameras)
    setup = dict(controller_version=parameters['controller_version'],
        stations={station: approved_station},
        source_kind='formal_workstation_and_storyboard',
        execution_review=copy.deepcopy(review))
    controller_factory=G2Controller
    if step == 4:
        from .state_g1_controller import StateG1Controller
        controller_factory=StateG1Controller
    elif step == 5:
        from .range_g1_controller import RangeG1Controller
        controller_factory=RangeG1Controller
    confirmed_reference=job.get('confirmed_sequence_reference_file')
    if confirmed_reference:
        if step != 5 or not mount_revision:
            raise ValueError('Confirmed sequence input requires the current reviewed-model measured-state executor')
        from .reviewed_sequence_controller import ReviewedSequenceController
        parameters['confirmed_sequence_reference_file']=str(Path(confirmed_reference).resolve())
        controller_factory=ReviewedSequenceController
    metadata = dict(entry_version=('v2_step5_preferred_contact_range_v1' if step == 5 else
                                   'v2_step4_measured_G1_phases_v2_preserved_approach' if step == 4 else
                                   ENTRY_VERSION if step == 1 else
                                   'v2_stage_aware_orientation_v1'),
        change_category=('data_input_only' if step == 1 else
                         'allowed_contact_radius_selection_only' if step == 5 else
                         'measured_phase_transitions_only' if step == 4 else
                         'stage_aware_working_side_and_rest_connection'),
        orientation_policy=POLICY_VERSION if step >= 2 else None,
        motion_rules_changed=step >= 2, score_changed=False,
        historical_orientation_conflicts_retained=step == 1,
        storyboard_file=str(inputs.storyboard_path),
        workstation_file=str(inputs.workstation_path),
        execution_review=copy.deepcopy(review),
        storyboard_snapshot=copy.deepcopy(inputs.storyboard),
        workstation_snapshot=copy.deepcopy(inputs.workstation))
    if model_binding is not None:
        metadata.update(entry_version='v2_reviewed_mount_and_free_connections_v1',
            change_category='reviewed_fixed_tool_mount_G1_conversion_and_free_connections',
            model_binding=model_binding,
            independent_return_target_changed=False,
            working_orientation_target_generator_changed=False,
            release_reference_changed=False)
    if parameters.get('release_reference_policy'):
        metadata.update(release_reference_changed=True,
            release_reference_policy=parameters['release_reference_policy'],
            change_category='G1_release_reference_only',
            independent_return_target_changed=False,
            working_orientation_target_generator_changed=False)
    if confirmed_reference:
        metadata.update(change_category='all_user_confirmed_sequence_references',
            confirmed_sequence_reference_file=parameters['confirmed_sequence_reference_file'],
            working_orientation_target_generator_changed=True,
            independent_return_target_changed=True,all_saved_intermediate_poses_consumed=True,
            old_default_controller_unchanged=True,score_changed=False)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return _run_prepared_episode(job, output, start=start,
        procedure_version=procedure_version, station=station, variant=inputs.operation,
        setup=setup, setup_path=inputs.workstation_path,
        source=inputs.storyboard_path, approved_station=approved_station,
        placement=inputs.placement, method=inputs.method,
        document={'keyposes': inputs.controller_poses}, parameters=parameters,
        offset=offset, scale=scale, controller_factory=controller_factory,
        workstation=inputs.physical_description, input_metadata=metadata,
        robot_adapter=selected_adapter,
        record=record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_episode(json.loads(args.job.read_text()), args.output, record=True)


if __name__ == '__main__':
    main()
