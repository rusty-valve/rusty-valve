"""Robot-only adapters and composition with an independent H0 workstation.

The environment and scorer never select a robot or know its part names. An
adapter supplies an MJCF file, its placement/control names and three mutually
exclusive collision-geometry categories. Built-in OpenArm examples reuse the
same upstream model and hook installer as bimanual_robot, without first making
and then deleting a valve scene. External MJCF + name-list documents use the
same public loader/composer; they need not select an OpenArm string.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
from typing import Mapping
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .geometry import _camera, _s
from .h0_scene import build_workstation_tree
from .h0_station import load_workstation

CONFIG_ROOT = Path(__file__).with_name('configs')/'robots'
ROLES = ('operation_end', 'idle_hand', 'robot_other')


@dataclass(frozen=True)
class RobotAdapter:
    _document: dict
    _directory: Path

    def __post_init__(self):
        d = self._document
        if d.get('schema_version') != 1 or not d.get('adapter_id'):
            raise ValueError('Robot adapter needs schema_version=1 and adapter_id')
        if not self.model_file.is_file():
            raise FileNotFoundError(self.model_file)
        if set(d['geometry_groups']) != set(ROLES):
            raise ValueError(f'Robot geometry_groups must declare exactly {ROLES}')
        if len(self.actuator_names) != len(self.controlled_joint_names):
            raise ValueError('actuator_names and controlled_joint_names must align')
        for name in ('actuator_names', 'controlled_joint_names', 'robot_joint_names'):
            values = getattr(self, name)
            if name == 'controlled_joint_names': values = tuple(x for x in values if x is not None)
            if len(values) != len(set(values)):
                raise ValueError(f'{name} contains duplicates')
        p = d['placement']
        if (len(p['base_position_m']) != 3 or not np.isfinite(p['base_position_m']).all()
                or not math.isfinite(float(p.get('base_yaw_deg', 0.)))):
            raise ValueError('Robot placement must be finite')
        if d['control']['mode'] not in ('joint_position', 'actuator_control'):
            raise ValueError('Control mode must be joint_position or actuator_control')

    @property
    def adapter_id(self): return self._document['adapter_id']
    @property
    def side(self): return self._document.get('active_side')
    @property
    def model_file(self):
        return (self._directory/Path(self._document['model_file'])).resolve()
    @property
    def actuator_names(self): return tuple(self._document['control']['actuator_names'])
    @property
    def controlled_joint_names(self):
        return tuple(self._document['control'].get('controlled_joint_names', [None]*len(self.actuator_names)))
    @property
    def robot_joint_names(self): return tuple(self._document['control']['robot_joint_names'])
    @property
    def active_joint_names(self): return tuple(self._document['control'].get('active_joint_names', ()))
    @property
    def active_actuator_names(self): return tuple(self._document['control'].get('active_actuator_names', ()))
    @property
    def idle_joint_names(self): return tuple(self._document['control'].get('idle_joint_names', ()))
    @property
    def idle_actuator_names(self): return tuple(self._document['control'].get('idle_actuator_names', ()))
    @property
    def ee_site_name(self): return self._document['control'].get('ee_site_name')
    @property
    def operation_body_name(self): return self._document['control'].get('operation_body_name')
    @property
    def has_idle_hand(self):
        group = self._document['geometry_groups']['idle_hand']
        return bool(group.get('body_names') or group.get('geom_names'))
    def to_dict(self):
        result = copy.deepcopy(self._document)
        result['model_file'] = str(self.model_file)
        return result
    def with_placement(self, *, base_position_xy_m=None, shoulder_height_m=None,
                       base_yaw_deg=None, base_position_m=None):
        document = self.to_dict(); p = document['placement']
        if base_position_m is not None:
            p['base_position_m'] = list(map(float, base_position_m))
            if 'shoulder_reference_offset_m' in p:
                p['shoulder_height_m'] = p['base_position_m'][2]+float(p['shoulder_reference_offset_m'])
        if base_position_xy_m is not None:
            if len(base_position_xy_m) != 2:
                raise ValueError('base_position_xy_m needs two numbers')
            p['base_position_m'][:2] = list(map(float, base_position_xy_m))
        if shoulder_height_m is not None:
            if 'shoulder_reference_offset_m' not in p:
                raise ValueError('This adapter has no declared shoulder height offset')
            p['shoulder_height_m'] = float(shoulder_height_m)
            p['base_position_m'][2] = float(shoulder_height_m)-float(p['shoulder_reference_offset_m'])
        if base_yaw_deg is not None:
            p['base_yaw_deg'] = float(base_yaw_deg)
        return RobotAdapter(document, self._directory)


def load_robot_adapter(name_or_path='openarm_s_hook'):
    if isinstance(name_or_path, RobotAdapter): return name_or_path
    if isinstance(name_or_path, Mapping):
        # A dictionary's relative model paths resolve against the working
        # directory; to_dict() writes an absolute path for unambiguous reuse.
        return RobotAdapter(copy.deepcopy(dict(name_or_path)), Path.cwd())
    path = Path(name_or_path)
    if not path.is_file(): path = CONFIG_ROOT/f'{name_or_path}.json'
    path = path.resolve()
    return RobotAdapter(json.loads(path.read_text(encoding='utf-8')), path.parent)


def _absolutize_assets(root, directory):
    compiler = root.find('compiler')
    if compiler is None: compiler = ET.SubElement(root, 'compiler')
    dirs = {tag: directory/compiler.get(f'{tag}dir', compiler.get('assetdir', ''))
            for tag in ('mesh', 'texture')}
    for node in root.findall('asset/*'):
        if node.get('file'):
            folder = dirs.get(node.tag, directory)
            node.set('file', str((folder/Path(node.get('file'))).resolve()))
    for key in ('meshdir', 'texturedir', 'assetdir'):
        compiler.attrib.pop(key, None)
    compiler.set('autolimits', 'true')


def _scope_robot_defaults(root):
    default = root.find('default')
    if default is None: default = ET.SubElement(root, 'default')
    old = list(default)
    for child in old: default.remove(child)
    scope = ET.SubElement(default, 'default', {'class': 'h0_robot_default'})
    scope.extend(old)
    world = root.find('worldbody')
    for node in world:
        if node.tag == 'body': node.attrib.setdefault('childclass', 'h0_robot_default')
        elif node.tag in ('geom', 'site', 'camera', 'light'):
            node.attrib.setdefault('class', 'h0_robot_default')
    for section in ('actuator', 'tendon', 'equality', 'contact'):
        for node in root.findall(f'{section}/*'):
            if node.tag != 'exclude': node.attrib.setdefault('class', 'h0_robot_default')


def _validate_preinstalled_s_hook(root):
    """Keep S-tool identity without installing a second copy into complete MJCF."""
    for side, site_name in (('left', 'left_ee_control'), ('right', 'ee_control')):
        hooks = root.findall(f".//body[@name='openarm_{side}_hook']")
        sites = root.findall(f".//site[@name='{site_name}']")
        if len(hooks) != 1 or len(sites) != 1:
            raise ValueError(f'Preinstalled S tool requires one {side} hook and control site')
        hook = hooks[0]
        if hook.find(f"site[@name='{site_name}']") is not sites[0]:
            raise ValueError(f'Preinstalled {side} S-tool control site must belong to its hook')
        parts = hook.findall('geom')
        names = {part.get('name', '') for part in parts}
        if not (any(name.startswith(f'{side}_hook_bend_0') for name in names) and
                {f'{side}_hook_bend_1', f'{side}_hook_bend_2'} <= names):
            raise ValueError(f'Preinstalled {side} S tool is missing its contact geometry')
        # Painted overlays share the name prefix but intentionally do not collide.
        contact_parts = [part for part in parts
                         if part.get('name', '').startswith(f'{side}_hook_bend_')
                         and part.get('material') == 's_hook_metal']
        if not contact_parts:
            raise ValueError(f'Preinstalled {side} S tool has no metal contact parts')
        if any(part.get('contype') == '0' and part.get('conaffinity') == '0'
               for part in contact_parts):
            raise ValueError(f'Preinstalled {side} S-tool contact geometry is non-colliding')
        if root.find(f".//body[@name='openarm_{side}_hand']") is not None:
            raise ValueError(f'Preinstalled {side} S tool still contains the stock hand')


def build_robot_tree(description='openarm_s_hook'):
    """Return independent robot MJCF, with no valve, pipe, fixture or floor."""
    adapter = load_robot_adapter(description); d = adapter.to_dict()
    source = adapter.model_file
    root = ET.parse(source).getroot()
    source_compiler = root.find('compiler')
    source_angle_unit = 'degree' if source_compiler is None else source_compiler.get('angle', 'degree')
    if root.find('.//include') is not None or source_angle_unit != 'radian':
        # Let the installed official MJCF parser flatten includes. Paths remain
        # relative to the submitted model's compiler asset directories. Also
        # canonicalize degree-based submissions before adding a radian task.
        root = ET.fromstring(mujoco.MjSpec.from_file(str(source)).to_xml())
    _absolutize_assets(root, source.parent)
    world = root.find('worldbody')
    base = world.find(f".//body[@name='{d['placement']['root_body']}']")
    if base is None: raise ValueError('Declared robot root_body does not exist')
    placement = d['placement']; position = list(map(float, placement['base_position_m']))
    base.set('pos', _s(position))
    yaw = math.radians(float(placement.get('base_yaw_deg', 0.)))
    original = np.asarray(placement.get('base_quaternion_wxyz', [1., 0., 0., 0.]), dtype=float)
    if original.shape != (4,) or np.linalg.norm(original) < 1e-10:
        raise ValueError('base_quaternion_wxyz must be a nonzero quaternion')
    original /= np.linalg.norm(original)
    orientation = np.zeros(4)
    mujoco.mju_mulQuat(orientation, np.array([math.cos(yaw/2), 0., 0., math.sin(yaw/2)]), original)
    base.set('quat', _s(orientation))
    # Optional changes belong to this robot adapter, never the workstation.
    variant = d.get('tool_adapter')
    preinstalled = d.get('tool_preinstalled', False)
    if not isinstance(preinstalled, bool) or (preinstalled and variant != 's_hook'):
        raise ValueError('tool_preinstalled requires tool_adapter=s_hook and a boolean flag')
    if variant == 's_hook':
        if preinstalled:
            _validate_preinstalled_s_hook(root)
        else:
            from .hook_tool import install_hook
            install_hook(root)
    elif variant not in (None, 'stock'):
        raise ValueError('Unknown tool_adapter; external complete MJCF needs no tool_adapter')
    for site in d.get('additional_sites', ()):
        parent = world.find(f".//body[@name='{site['parent_body']}']")
        if parent is None: raise ValueError('Additional site parent is missing')
        if root.find(f".//site[@name='{site['name']}']") is None:
            ET.SubElement(parent, 'site', name=site['name'], pos=_s(site['position_local_m']),
                          size='.001', rgba='0 0 0 0', group='4')
    for camera in d.get('cameras', ()):
        parent = world.find(f".//body[@name='{camera['parent_body']}']")
        if parent is None: raise ValueError('Camera parent is missing')
        start = np.asarray(camera['position_local_m'], dtype=float)
        direction = np.asarray(camera['forward_direction_local'], dtype=float)
        _camera(parent, camera['name'], start, start+direction, camera['vertical_fov_deg'])
    if d.get('camera_rig'):
        from .camera_rig import install_camera_rig
        install_camera_rig(root, d)
    if d.get('mount_support'):
        x, y, z = position
        if z <= 0: raise ValueError('A floor pedestal requires positive mounting height')
        ET.SubElement(world, 'geom', name='robot_mount_stand', type='box',
            pos=_s((x, y, z/2)), size=_s((.075, .09, z/2)), quat=_s(orientation),
            rgba='.17 .20 .23 1')
        ET.SubElement(world, 'geom', name='robot_mount_foot', type='box',
            pos=_s((x, y, .018)), size='.19 .22 .018', quat=_s(orientation),
            rgba='.17 .20 .23 1')
    _scope_robot_defaults(root)
    root.set('model', f'H0_robot_{adapter.adapter_id}')
    return root


def _subtree_bodies(model, names):
    result = set()
    for name in names:
        result.add(int(model.body(name).id))
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in result: result.add(body)
    return result


def _collision_geoms(model):
    result = set(np.flatnonzero((model.geom_contype != 0)|(model.geom_conaffinity != 0)).tolist())
    result.update(map(int, model.pair_geom1)); result.update(map(int, model.pair_geom2))
    return result


def robot_geom_groups(model, description='openarm_s_hook'):
    """Resolve exclusive role categories; names and exclusions stay adapter-side."""
    adapter = load_robot_adapter(description); d = adapter.to_dict()
    body_ids = _subtree_bodies(model, d['robot_scope'].get('root_bodies', ()))
    scope = {int(i) for i in range(model.ngeom) if int(model.geom_bodyid[i]) in body_ids}
    scope.update(int(model.geom(name).id) for name in d['robot_scope'].get('geom_names', ()))
    scope &= _collision_geoms(model)
    if not scope: raise ValueError('Robot adapter scope contains no collision geometry')
    groups = {}
    for role in ROLES:
        rule = d['geometry_groups'][role]
        if rule.get('remaining_robot_collision_geoms'):
            if role != 'robot_other': raise ValueError('Only robot_other may select the remainder')
            continue
        bodies = _subtree_bodies(model, rule.get('body_names', ()))
        ids = {int(i) for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies}
        ids.update(int(model.geom(name).id) for name in rule.get('geom_names', ()))
        groups[role] = ids & _collision_geoms(model)
        if not groups[role] <= scope:
            raise ValueError(f'{role} selects geometry outside the declared robot scope')
    if 'robot_other' not in groups:
        groups['robot_other'] = scope-groups['operation_end']-groups['idle_hand']
    if not groups['operation_end']:
        raise ValueError('An adapter must declare at least one operation_end collision geometry')
    if any(groups[a]&groups[b] for i, a in enumerate(ROLES) for b in ROLES[i+1:]):
        raise ValueError('Robot role geometry groups overlap')
    if set.union(*groups.values()) != scope:
        raise ValueError('Robot role geometry groups do not cover all robot collision geometry')
    return {role: sorted(groups[role]) for role in ROLES} | dict(
        robot_geom_ids=sorted(scope), has_idle_hand=bool(groups['idle_hand']))


def _actuator_reference(model, names):
    return {name: dict(ctrlrange=model.actuator_ctrlrange[model.actuator(name).id].tolist(),
        forcerange=model.actuator_forcerange[model.actuator(name).id].tolist(),
        gainprm=model.actuator_gainprm[model.actuator(name).id].tolist(),
        biasprm=model.actuator_biasprm[model.actuator(name).id].tolist(),
        ctrllimited=bool(model.actuator_ctrllimited[model.actuator(name).id]),
        forcelimited=bool(model.actuator_forcelimited[model.actuator(name).id])) for name in names}


def _prune_temporary_display_geometry(root):
    """Prune only demonstrably mass-neutral display geoms in a private tree.

    The real scene tree never enters this function. Conservatively leave
    unfamiliar/include/plugin/deformable models to the original compiler.
    Explicit contact and sensor references protect their named geometry even
    when its ordinary contact masks are zero.
    """
    if any(root.find(f'.//{tag}') is not None for tag in
           ('include', 'plugin', 'extension', 'deformable', 'flexcomp', 'composite')):
        return False
    compiler = root.find('compiler')
    if compiler is not None and compiler.get('inertiafromgeom', 'auto') == 'true':
        # In this mode explicit inertial elements do not prove mass neutrality.
        return False

    def zero(value):
        try:
            return value is not None and float(value) == 0.
        except (TypeError, ValueError):
            return False

    visual_defaults = root.findall(".//default[@class='visual']/geom")
    visual_class_safe = (len(visual_defaults) == 1
        and zero(visual_defaults[0].get('contype'))
        and zero(visual_defaults[0].get('conaffinity')))
    # An explicit mass overrides density. If any default supplies nonzero
    # mass, density=0 alone is insufficient evidence without resolving that
    # model's complete default hierarchy; simply retain those geoms.
    defaults_have_mass = any(g.get('mass') is not None and not zero(g.get('mass'))
                             for g in root.findall('.//default/geom'))
    protected = set()
    for node in root.iter():
        if node.tag != 'geom':
            protected.update(node.get(key) for key in ('geom', 'geom1', 'geom2') if node.get(key))
        if node.get('objtype') == 'geom' and node.get('objname'):
            protected.add(node.get('objname'))
        if node.get('reftype') == 'geom' and node.get('refname'):
            protected.add(node.get('refname'))
    for body in list(root.iter('body')) + list(root.iter('worldbody')):
        explicit_inertia = body.find('inertial') is not None
        for geom in list(body.findall('geom')):
            if geom.get('name') in protected:
                continue
            # A geom may override a zero-mask visual class with live collision
            # masks. Both effective masks must still be explicitly known zero.
            class_is_visual = geom.get('class') == 'visual' and visual_class_safe
            default_masks = visual_defaults[0] if class_is_visual else {}
            display_only = (zero(geom.get('contype', default_masks.get('contype')))
                            and zero(geom.get('conaffinity', default_masks.get('conaffinity'))))
            mass_neutral = (explicit_inertia or zero(geom.get('mass')) or
                (geom.get('mass') is None and not defaults_have_mass and zero(geom.get('density'))))
            if display_only and mass_neutral:
                body.remove(geom)
    used_meshes = {geom.get('mesh') for geom in root.iter('geom') if geom.get('mesh')}
    asset = root.find('asset')
    if asset is not None:
        for mesh in list(asset.findall('mesh')):
            if mesh.get('name') not in used_meshes:
                asset.remove(mesh)
    return True


def _compile_temporary_validation_model(*, xml=None, source_path=None):
    """Compile a private reduced tree for checks; keep final full XML intact."""
    if (xml is None) == (source_path is None):
        raise ValueError('Supply exactly one temporary XML string or source path')
    if source_path is not None:
        source_path = Path(source_path)
        original_compile = lambda: mujoco.MjModel.from_xml_path(str(source_path))
        try:
            tree = ET.parse(source_path).getroot()
        except (OSError, ET.ParseError):
            return original_compile()
    else:
        original_compile = lambda: mujoco.MjModel.from_xml_string(xml)
        try:
            tree = ET.fromstring(xml)
        except ET.ParseError:
            return original_compile()
    if not _prune_temporary_display_geometry(tree):
        return original_compile()
    if source_path is not None:
        compiler = tree.find('compiler')
        # strippath changes asset lookup semantics; keep the established loader
        # for that unusual source instead of guessing a rewritten asset path.
        if compiler is not None and compiler.get('strippath', 'false') != 'false':
            return original_compile()
        autolimits = None if compiler is None else compiler.get('autolimits')
        _absolutize_assets(tree, source_path.parent)
        compiler = tree.find('compiler')
        if autolimits is None:
            compiler.attrib.pop('autolimits', None)
        else:
            compiler.set('autolimits', autolimits)
    # Unsupported dependencies on a removed visual/mesh must not weaken a
    # validation check. Let the unchanged original loader handle such inputs.
    try:
        return mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
    except ValueError:
        return original_compile()


def compose_scene(workstation='A1', robot_adapter='openarm_s_hook', *, return_compiled=False):
    """Return (MJCF XML, metadata), optionally with the validated full model.

    ``return_compiled=True`` returns ``(xml, metadata, model)`` so an environment
    can own the same fully compiled model that passed the checks below. The
    default two-item return remains compatible with XML-only callers. Temporary
    validation compilations can omit proven mass-neutral display geometry;
    returned XML and the actual model always retain the full scene.
    """
    station = load_workstation(workstation); adapter = load_robot_adapter(robot_adapter)
    region = station.to_dict()['allowed_base_region_world']
    base_xy = np.asarray(adapter.to_dict()['placement']['base_position_m'][:2], dtype=float)
    if (np.any(base_xy < np.asarray(region['xy_min_m']))
            or np.any(base_xy > np.asarray(region['xy_max_m']))):
        raise ValueError('Robot base reference lies outside the workstation allowed region')
    root = build_robot_tree(adapter)
    robot_xml = ET.tostring(root, encoding='unicode')
    robot_model = _compile_temporary_validation_model(xml=robot_xml)
    groups_before = robot_geom_groups(robot_model, adapter)
    if set(groups_before['robot_geom_ids']) != _collision_geoms(robot_model):
        raise ValueError('Robot scope omits collision geometry present in the submitted robot MJCF')
    if set(adapter.actuator_names) != {robot_model.actuator(i).name for i in range(robot_model.nu)}:
        raise ValueError('Adapter must declare all robot actuators, once each')
    if set(adapter.robot_joint_names) != {robot_model.joint(i).name for i in range(robot_model.njnt)}:
        raise ValueError('Adapter must declare all robot joints, including passive coupled joints')
    # Comparing by declared actuator name also handles removed gripper motors.
    source_model = _compile_temporary_validation_model(source_path=adapter.model_file)
    reference = _actuator_reference(source_model, adapter.actuator_names)
    # The reference contains only plain lists/bools. Do not keep a second
    # complete mesh model alive during subsequent scene compilation.
    del source_model
    gc.collect()
    if _actuator_reference(robot_model, adapter.actuator_names) != reference:
        raise ValueError('Robot adaptation changed reference actuator gains or limits')
    d = adapter.to_dict(); home_name = d.get('initial_keyframe')
    robot_data = mujoco.MjData(robot_model)
    if home_name:
        mujoco.mj_resetDataKeyframe(robot_model, robot_data, robot_model.key(home_name).id)
    # A free root's old keyframe pose must not silently undo declared placement.
    root_id = int(robot_model.body(d['placement']['root_body']).id)
    for jid in range(robot_model.njnt):
        if int(robot_model.jnt_bodyid[jid]) == root_id and int(robot_model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            adr = int(robot_model.jnt_qposadr[jid])
            robot_data.qpos[adr:adr+7] = robot_model.qpos0[adr:adr+7]
    initial_robot_qpos = robot_data.qpos.copy(); initial_robot_ctrl = robot_data.ctrl.copy()
    root.remove(root.find('keyframe')) if root.find('keyframe') is not None else None
    environment = build_workstation_tree(station)
    if environment.find('actuator') is not None and len(environment.find('actuator')):
        raise ValueError('Workstation must remain passive and robot-independent')
    env_default = environment.find('default')
    if root.find("default/default[@class='valve_task']") is not None:
        raise ValueError('Robot model uses reserved environment default class valve_task')
    root.find('default').extend(copy.deepcopy(list(env_default)))
    for section in ('asset', 'worldbody', 'contact', 'equality', 'sensor', 'tendon'):
        incoming = environment.find(section)
        if incoming is None: continue
        outgoing = root.find(section)
        if outgoing is None: outgoing = ET.SubElement(root, section)
        outgoing.extend(copy.deepcopy(list(incoming)))
    option = root.find('option')
    if option is None: option = ET.SubElement(root, 'option')
    option.attrib.update(environment.find('option').attrib)
    visual = root.find('visual')
    if visual is None: visual = ET.SubElement(root, 'visual')
    for node in list(environment.find('visual') or ()):
        old = visual.find(node.tag)
        if old is None: visual.append(copy.deepcopy(node))
        else: old.attrib.update(node.attrib)
    global_settings = visual.find('global')
    if global_settings is None: global_settings = ET.SubElement(visual, 'global')
    global_settings.set('offwidth', '1440');global_settings.set('offheight', '1000')
    root.set('model', f'H0_{station.case}_{adapter.adapter_id}')
    # The environment is appended after the complete robot: rebuild keys at
    # compiled dimensions instead of assuming fingers and actuators align.
    provisional = _compile_temporary_validation_model(xml=ET.tostring(root, encoding='unicode'))
    keyframes = ET.SubElement(root, 'keyframe')
    defaults = provisional.qpos0.copy()
    defaults[:robot_model.nq] = initial_robot_qpos
    valve_adr = int(provisional.joint(station.valve_joint_name).qposadr[0])
    # Keyframe construction below needs these owned arrays and scalar address,
    # not the provisional model's mesh buffers.
    del provisional
    gc.collect()
    for name, angle in (('h0_open', station.q_open_rad), ('h0_closed', station.q_close_rad)):
        values = defaults.copy();values[valve_adr] = angle
        ET.SubElement(keyframes, 'key', name=name, qpos=_s(values), ctrl=_s(initial_robot_ctrl))
    # Human inspection is a separate, fixed camera across the pipe. Its frame
    # comes from the declared setup, never from the future action trajectory.
    from .observer_camera import attach_observer_camera, OBSERVER_CAMERA_NAME
    if any(c.get('name') == OBSERVER_CAMERA_NAME for c in d.get('cameras', ())):
        raise ValueError(f'{OBSERVER_CAMERA_NAME} is reserved for workstation observation')
    shoulder_height = d['placement'].get('shoulder_height_m')
    height_source = 'robot adapter shoulder_height_m'
    if shoulder_height is None:
        # Generic robots need no OpenArm names or new mandatory adapter field.
        # Use their reset-pose collision geometry to estimate work height.
        mujoco.mj_kinematics(robot_model, robot_data)
        robot_ids = np.asarray(groups_before['robot_geom_ids'], dtype=int)
        shoulder_height = float(np.max(robot_data.geom_xpos[robot_ids, 2]
                                       + robot_model.geom_rbound[robot_ids]))
        height_source = 'upper bound of declared robot collision geometry at reset'
    robot_nq, robot_nv = int(robot_model.nq), int(robot_model.nv)
    # The generic shoulder-height fallback is the last use of robot kinematics.
    # Release data first (it references the model), before compiling the final
    # combined scene. Geometry, keyframes and all validation stay unchanged.
    del robot_data, robot_model
    gc.collect()
    observer = attach_observer_camera(root, station, adapter,
                                      shoulder_height_m=shoulder_height)
    observer['work_height_source'] = height_source
    if d.get('camera_rig'):
        from .camera_rig import camera_metadata
        cameras = camera_metadata(d, root=root) + [observer]
    else:
        cameras = copy.deepcopy(d.get('cameras', [])) + [observer]
    ET.indent(root)
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    reference_after = _actuator_reference(model, adapter.actuator_names)
    if reference_after != reference: raise ValueError('Composition changed reference actuator limits')
    # A controlled-joint list maps controls rather than all qpos: coupled
    # passive finger joints stay in robot_joint_names but have no fake motors.
    for actuator_name, joint_name in zip(adapter.actuator_names, adapter.controlled_joint_names):
        if joint_name is None:
            if d['control']['mode'] == 'joint_position':
                raise ValueError('joint_position control must name each scalar transmission joint')
            continue  # Actuator-control submissions may use tendon/site transmissions.
        actuator = model.actuator(actuator_name)
        if int(model.actuator_trnid[actuator.id, 0]) != int(model.joint(joint_name).id):
            raise ValueError('Declared actuator-to-joint mapping does not match the MJCF')
    groups = robot_geom_groups(model, adapter)
    metadata = dict(adapter=adapter.to_dict(), adapter_id=adapter.adapter_id,
        station=station.case, initial_robot_qpos=initial_robot_qpos.tolist(),
        initial_robot_ctrl=initial_robot_ctrl.tolist(), robot_nq=robot_nq,
        robot_nv=robot_nv, robot_geom_groups=groups, cameras=cameras,
        original_actuator_limits=reference, actuator_limits_unchanged=True,
        force_limit_reference=str(adapter.model_file),
        robot_collision_geometry_count=len(groups_before['robot_geom_ids']),
        keyframes=dict(open='h0_open', closed='h0_closed'),
        note='Independent passive workstation + declared robot. No driving actuator added to valve.')
    return (xml, metadata, model) if return_compiled else (xml, metadata)


compose_workstation = compose_scene
