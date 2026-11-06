"""Explicit 20/35 cm stance experiment; ordinary approved G2 stays frozen.

Validation reads JSON only. The CLI defaults to validation and never starts
physics unless --run is explicit and the new key-pose review is confirmed.
The original approved G1 files remain reference material, not approval of a
different robot placement.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "nominal_20_35cm_r175_v21"
RADIUS_M = .175


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path(value, root):
    path = Path(value)
    return (root/path if not path.is_absolute() else path).resolve()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _source(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must identify its source")
    return value


def validate_job_context(job, root=ROOT, *, require_execution_approval=False):
    """Read-only validation, also safe for a plan before user image review.

    Distances are to the NOMINAL shaft along body-forward horizontal X. Valve
    randomization does not translate the base, head mount or world observer.
    """
    root = Path(root).resolve()
    if job.get("execution_profile") != PROFILE:
        raise ValueError(f"This experimental entry requires execution_profile={PROFILE}")
    station, variant = job.get("station"), job.get("variant", job.get("operation"))
    if station not in ("A1", "A2") or variant not in ("close", "open", "slow_open"):
        raise ValueError("The stance experiment is limited to A1/A2 close/open/slow_open")
    task = "close" if variant == "close" else "open"
    setup_path = _path(job.get("approved_setup", "valve_bench/configs/g2_approved_auto.json"),root)
    setup = _read(setup_path)
    if setup.get("user_approved") is not True or setup.get("frozen_for_G2") is not True:
        raise ValueError("The original reference setup must remain approved and frozen")
    reviewed_version=job.get('approved_path_mode')
    reviewed_path_mode=reviewed_version in ('a2_reviewed_forehand_path_v1','a2_reviewed_forehand_path_v2_seated35','a2_forehand_v3_seated_continuous')
    expected_controller=reviewed_version if reviewed_path_mode else 'approved_g1_auto_v3'
    if setup.get("controller_version") != expected_controller:
        raise ValueError("Unexpected original reference controller")
    reference = setup["stations"][station]
    expected_source = _path(reference["accepted_keyposes"][task],root)
    source = _path(job.get("keyposes",expected_source),root)
    if source != expected_source:
        raise ValueError("Use the original approved G1 document as the reference")
    document = _read(source)
    reference_placement = copy.deepcopy(reference["placement"])
    if document["placement"] != reference_placement:
        raise ValueError("Original G1 reference and frozen reference setup disagree")
    if job.get("approved_placement",reference_placement) != reference_placement:
        raise ValueError("approved_placement is a historical reference, not an experiment override")
    if document["method"].get("station") != station or document["method"].get("task") != task or document["method"].get("side") != "left":
        raise ValueError("Original left-hand reference method does not match this job")
    for key,value in reference["methods"][task].items():
        if document["method"].get(key) != value:
            raise ValueError(f"Original reference method changed: {key}")
    distance = _number(job.get("nominal_chest_distance_m"),"nominal_chest_distance_m")
    if distance not in (.20,.35):
        raise ValueError("Only nominal chest distances .20 and .35 m are in this experiment")
    reviewed_distance=.35 if reviewed_version in ('a2_reviewed_forehand_path_v2_seated35','a2_forehand_v3_seated_continuous') else .20
    if reviewed_path_mode and (station!='A2' or variant not in ('close','open') or distance!=reviewed_distance):
        raise ValueError('Reviewed forehand requires A2 open/close at its matching frozen placement')
    if reviewed_path_mode and not document.get('approved_path'):
        raise ValueError('The reviewed forehand approach, contact and exit paths are required')
    if reviewed_version=='a2_forehand_v3_seated_continuous' and require_execution_approval:
        review=document.get('execution_approval',{})
        if (review.get('approved') is not True or not review.get('candidate_id')
                or review.get('scope')!='complete_approach_contact_exit'
                or review.get('controller_version')!=reviewed_version):
            raise ValueError('New small-tilt candidates and the complete approach/contact/exit preview must be confirmed first')
        if job.get('procedure_version')!='A1-v2.4':
            raise ValueError('The repaired A2 controller requires the shared v2.4 score and full evidence log')
    params = job.get("parameters",job.get("candidate",{}))
    radius = _number(params.get("r_m",params.get("r")),"parameters.r_m")
    expected_radius=RADIUS_M
    if reviewed_version=='a2_forehand_v3_seated_continuous':
        from .contact_target import target_radius,load_contact_target_rule
        target_spec=_read(root/f"valve_bench/configs/workstations/{station}.json")
        expected_radius=target_radius(target_spec['valve']['handle_length_m'])
        if document.get('target_contact_fraction')!=load_contact_target_rule()['target_contact_fraction']:
            raise ValueError('The reviewed contact fraction differs from the current user-defined rule')
    if abs(radius-expected_radius)>1e-12:
        raise ValueError("Contact radius must match the reviewed handle-length fraction (historical plans retain their frozen radius)")
    authorization = _source(job.get("experiment_authorization_source"),"experiment_authorization_source")
    revision = _source(job.get("experiment_revision"),"experiment_revision")
    approval = copy.deepcopy(job.get("execution_approval",{}))
    if approval.get("state") not in ("pending_keypose_review","confirmed"):
        raise ValueError("execution_approval.state must be pending_keypose_review or confirmed")
    _source(approval.get("source"),"execution_approval.source")
    if require_execution_approval and approval["state"] != "confirmed":
        raise ValueError("New stance key-pose review is still pending; only preparation is authorized")
    spec = _read(root/f"valve_bench/configs/workstations/{station}.json")
    nominal_axis = list(map(float,spec["valve"]["axis_origin_world_m"]))
    reference_distance = _number(reference["stance"]["chest_to_valve_axis_m"],"reference chest distance")
    if reference_placement["base_yaw_deg"] != 0:
        raise ValueError("The original body yaw must be zero")
    expected_placement = copy.deepcopy(reference_placement)
    if not reviewed_path_mode:
        expected_placement["base_position_xy_m"][0] += reference_distance-distance
    elif abs(reference_distance-distance)>1e-10:
        raise ValueError('Reviewed forehand distance must match the fixed approved placement')
    execution = copy.deepcopy(job.get("execution_placement"))
    if not isinstance(execution,dict) or set(execution) != set(expected_placement):
        raise ValueError("execution_placement must retain the original placement fields")
    xy = execution.get("base_position_xy_m")
    if not isinstance(xy,(list,tuple)) or len(xy)!=2:
        raise ValueError("execution_placement needs two base coordinates")
    for i,coordinate in enumerate(xy):
        actual = _number(coordinate,f"execution_placement.xy[{i}]")
        if abs(actual-expected_placement["base_position_xy_m"][i])>1e-10:
            raise ValueError("Only nominal front/back distance may change; original station lateral stays fixed")
    for key in ("base_yaw_deg","shoulder_height_m"):
        if _number(execution.get(key),f"execution_placement.{key}") != reference_placement[key]:
            raise ValueError(f"The experiment must retain {key}")
    region = spec["allowed_base_region_world"]
    if any(not low <= value <= high for value,low,high in zip(xy,region["xy_min_m"],region["xy_max_m"])):
        raise ValueError("Experimental base is outside the workstation's allowed region")
    offset = job.get("offset_m",[0.,0.,0.])
    if not isinstance(offset,(list,tuple)) or len(offset)!=3:
        raise ValueError("offset_m must have three coordinates")
    offset = [_number(value,"offset_m") for value in offset]
    return dict(execution_profile=PROFILE,experiment_revision=revision,
        experiment_authorization_source=authorization,execution_approval=approval,
        experiment_placement_is_original_G1_approved=reviewed_path_mode,
        approved_reference_setup=str(setup_path),approved_reference_keyposes=str(source),
        approved_reference_placement=reference_placement,actual_execution_placement=execution,
        reference_nominal_chest_distance_m=reference_distance,nominal_chest_distance_m=distance,
        actual_chest_distance_to_randomized_axis_m=distance+offset[0],
        nominal_valve_axis_world_m=nominal_axis,
        torso_front_offset_from_base_m=nominal_axis[0]-reference_placement["base_position_xy_m"][0]-reference_distance,
        distance_measurement="horizontal body-forward projection from upper-torso visual front to nominal valve shaft axis",
        base_policy="chosen before reset from nominal station; fixed throughout; does not follow randomized valve",
        keypose_use=("reviewed forehand paths drive free motion and measured-hinge contact feedback; fixed approved placement"
                     if reviewed_path_mode else "unchanged approved document supplies method and IK seeds; new placement is experimental"),
        camera_policy="unchanged head local XML and FOV; unchanged observer world XML",
        context_validation_physics_steps_run=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--record",action="store_true")
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    job=_read(args.job)
    context=validate_job_context(job,require_execution_approval=args.run)
    if not args.run:
        print(json.dumps(context,ensure_ascii=False,indent=2,allow_nan=False))
        return
    if args.output is None:
        parser.error("--run requires --output")
    from .g2_episode import run_episode
    run_episode(job,args.output,record=args.record)


if __name__ == "__main__":
    main()
