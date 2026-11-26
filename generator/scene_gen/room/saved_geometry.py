"""Read shapes and meshes from a saved room for inspection."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2]
DISPLAY=PROJECT/'valve_bench/.venv-pose/lib/python3.12/site-packages'


def dependencies():
    from scene_gen.room.evaluation_common import require_environment
    environment=require_environment()
    global np,mujoco,viser,trimesh
    import numpy as np
    import mujoco
    before={'prefix':str(Path(sys.prefix).resolve()),'mujoco':str(Path(mujoco.__file__).resolve()),'numpy':str(Path(np.__file__).resolve()),'version':mujoco.__version__}
    if not all(Path(before[k]).is_relative_to(Path(before['prefix'])) for k in ('mujoco','numpy')):
        raise SystemExit('MuJoCo and NumPy are outside the expected environment')
    appended=False
    try:
        import viser
        import trimesh
    except ModuleNotFoundError:
        if not DISPLAY.is_dir():raise SystemExit('The existing read-only viewer library is unavailable')
        if str(DISPLAY) not in sys.path:sys.path.append(str(DISPLAY));appended=True
        import viser
        import trimesh
    require_environment()
    after={'prefix':str(Path(sys.prefix).resolve()),'mujoco':str(Path(mujoco.__file__).resolve()),'numpy':str(Path(np.__file__).resolve()),'version':mujoco.__version__}
    if before!=after:raise SystemExit('The viewer library changed the runtime source')
    for module in (viser,trimesh):
        path=Path(module.__file__).resolve()
        if not any(path.is_relative_to(root) for root in (Path(before['prefix']),DISPLAY)):
            raise SystemExit('Viewer library was loaded from outside its permitted read-only location')
    return {**environment,'locked_engine':before,'engine_unchanged_after_display_import':True,
        'display_path_append_only':appended,'display_directory':str(DISPLAY),'site_addsitedir_or_pth_used':False,
        'viser':{'version':viser.__version__,'path':str(Path(viser.__file__).resolve())},
        'trimesh':{'version':trimesh.__version__,'path':str(Path(trimesh.__file__).resolve())},'installed_or_modified_packages':False}


def geometry_dependencies():
    from .evaluation_common import PORTABLE_GENERATION, require_environment
    if not PORTABLE_GENERATION.get():
        return dependencies()
    require_environment()
    global np, mujoco, trimesh
    import numpy as np
    import mujoco
    import trimesh


def stamp(path):
    path=Path(path);s=path.stat()
    return {'path':str(path.resolve()),'mtime_ns':s.st_mtime_ns,'size_bytes':s.st_size}


def unchanged(records):
    if any(stamp(r['path'])!=r for r in records):raise ValueError('Original model or JSON changed during viewing')


def mesh(model,gid,cache):
    kind=int(model.geom_type[gid]);size=model.geom_size[gid]
    key=(kind,int(model.geom_dataid[gid])) if kind==int(mujoco.mjtGeom.mjGEOM_MESH) else (kind,*size.tolist())
    if key in cache:return cache[key]
    if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
        mid=int(model.geom_dataid[gid]);va=int(model.mesh_vertadr[mid]);vn=int(model.mesh_vertnum[mid]);fa=int(model.mesh_faceadr[mid]);fn=int(model.mesh_facenum[mid])
        result=np.asarray(model.mesh_vert[va:va+vn],float),np.asarray(model.mesh_face[fa:fa+fn],int)
    else:
        if kind==int(mujoco.mjtGeom.mjGEOM_BOX):m=trimesh.creation.box(extents=2*size)
        elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):m=trimesh.creation.cylinder(radius=size[0],height=2*size[1],sections=48)
        elif kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE):m=trimesh.creation.capsule(radius=size[0],height=2*size[1],count=[16,16])
        elif kind in (int(mujoco.mjtGeom.mjGEOM_SPHERE),int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)):
            m=trimesh.creation.icosphere(subdivisions=2);m.vertices*=size[0] if kind==int(mujoco.mjtGeom.mjGEOM_SPHERE) else size
        else:raise ValueError('The actual geom type cannot be displayed: '+str(model.geom(gid).name))
        result=np.asarray(m.vertices),np.asarray(m.faces)
    vertices,faces=result
    if not len(vertices) or not len(faces) or faces.min()<0 or faces.max()>=len(vertices) or not np.isfinite(vertices).all():raise ValueError('The actual mesh is missing')
    cache[key]=result
    return result
