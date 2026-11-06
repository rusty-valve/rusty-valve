"""Control helpers for the opt-in A2 v3 controller; no physical stepping.

The recess test describes the finite S tool and the existing handle section.
It is a controller seating state, not a new score, clearance or force limit.
Continuous IK uses a local numerical trust region so a better tool fit cannot
replace the arm posture with a distant joint branch in one command.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.spatial.transform import Rotation

from .hook_tool import tool_geometry


def _inside_triangle(point, triangle):
    edges=np.roll(triangle,-1,axis=0)-triangle
    delta=np.asarray(point)-triangle
    cross=edges[:,0]*delta[:,1]-edges[:,1]*delta[:,0]
    # Roundoff only, not a physical clearance allowance.
    return bool(np.all(cross>=-1e-14) or np.all(cross<=1e-14))


def _clip_polygon(polygon, normal, threshold):
    """Clip a physical section by n.dot(p) >= threshold (roundoff only)."""
    if not len(polygon):return []
    result=[]
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        da=float(normal@a-threshold);db=float(normal@b-threshold)
        if da>=-1e-14:result.append(a)
        if (da>=0.)!=(db>=0.):result.append(a+(b-a)*da/(da-db))
    return result


def recess_seating_geometry(*, tool_origin, tool_rotation, pivot, radial,
        stem, mode, target_radius_m, handle_length_m, handle_width_m,
        handle_thickness_m):
    """Check the actual handle axis in the real orange/blue recess.

    The requested actual handle cross-section must enter the relevant finite
    concavity. With a slanted tool, its centreline need not pass through the
    width mid-plane of the tool. We do NOT require the whole handle section
    to cross a mouth line, simultaneous force on both flanks, or a millimetre
    pose tolerance.
    Solved current working-face contact is checked separately by the caller.
    """
    if mode not in ('push','pull'):
        raise ValueError('Expected push or pull')
    cfg,_,plus,minus,_=tool_geometry()
    rotation=np.asarray(tool_rotation,float)
    radial=np.asarray(radial,float);stem=np.asarray(stem,float)
    local_pivot=rotation.T@(np.asarray(pivot,float)-np.asarray(tool_origin,float))
    local_radial=rotation.T@radial
    result=dict(in_recess=False,target_section_overlaps_tool=False,
        geometry_seated=False,target_radius_m=float(target_radius_m),
        basis='actual_handle_axis_and_finite_recess',mode=mode)
    triangle=np.asarray(minus[:3] if mode=='push' else plus[1:4])
    target=local_pivot+float(target_radius_m)*local_radial
    tangent=np.cross(stem,radial)
    half_section_x=(handle_width_m*.5*abs((rotation.T@tangent)[0])+
                    handle_thickness_m*.5*abs((rotation.T@stem)[0]))
    half_tool_width=cfg['out_of_plane_width_mm']*.0005
    overlaps=abs(target[0])<=half_tool_width+half_section_x
    half_tangent=rotation.T@tangent*handle_width_m*.5
    half_stem=rotation.T@stem*handle_thickness_m*.5
    polygon=[target-half_tangent-half_stem,target+half_tangent-half_stem,
             target+half_tangent+half_stem,target-half_tangent+half_stem]
    polygon=_clip_polygon(polygon,np.array([1.,0.,0.]),-half_tool_width)
    polygon=_clip_polygon(polygon,np.array([-1.,0.,0.]),-half_tool_width)
    first=triangle[1]-triangle[0];second=triangle[2]-triangle[0]
    orientation=first[0]*second[1]-first[1]*second[0]
    orientation=1. if orientation>0 else -1.
    for a,b in zip(triangle,np.roll(triangle,-1,axis=0)):
        edge=b-a;normal=orientation*np.array([0.,-edge[1],edge[0]])
        polygon=_clip_polygon(polygon,normal,float(normal@np.r_[0.,a]))
    area=0.
    if len(polygon)>=3:
        area=.5*sum(np.linalg.norm(np.cross(polygon[i]-polygon[0],polygon[i+1]-polygon[0]))
                    for i in range(1,len(polygon)-1))
    in_recess=bool(0.<target_radius_m<handle_length_m and area>1e-16)
    result.update(in_recess=bool(in_recess),target_section_overlaps_tool=bool(overlaps),
        geometry_seated=bool(in_recess and overlaps),
        target_section_tool_local_m=target.tolist(),
        target_section_inside_recess_area_m2=float(area),
        reason='seated_geometry' if in_recess and overlaps else 'handle_not_seated_at_target_section')
    return result


def bounded_continuation_solve(ik, position, rotation, seed, iterations, *, step_rad=.05):
    """One local IK continuation, with no remote seed or branch replacement.

    step_rad is an internal solver trust-region size, never an exam threshold.
    All native bounds are restored even if solving fails. Residuals are kept;
    being unable to reach a target is not silently relabelled as success.
    """
    native=ik.bounds.copy()
    seed=np.clip(np.asarray(seed,float),native[:,0]+1e-5,native[:,1]-1e-5)
    local=native.copy()
    local[:,0]=np.maximum(local[:,0],seed-step_rad)
    local[:,1]=np.minimum(local[:,1],seed+step_rad)
    try:
        ik.bounds[:]=local
        return ik.solve(position,rotation,seed,iterations)
    finally:
        ik.bounds[:]=native


class ContinuousJointPath:
    """Shape-preserving joint interpolation, stationary at both endpoints."""
    def __init__(self, coordinates, joints, errors=None):
        self.x=np.asarray(coordinates,float)
        self.joints=np.asarray(joints,float)
        if (len(self.x)<2 or self.joints.shape!=(len(self.x),7) or
                not np.all(np.diff(self.x)>0) or not np.isfinite(self.joints).all()):
            raise ValueError('Finite seven-joint path with increasing coordinates required')
        derivatives=PchipInterpolator(self.x,self.joints,axis=0).derivative()(self.x)
        derivatives[0]=0.;derivatives[-1]=0.
        self.curve=CubicHermiteSpline(self.x,self.joints,derivatives,axis=0)
        self.errors=np.zeros((len(self.x),2)) if errors is None else np.asarray(errors,float)

    def at(self, coordinate):
        x=float(np.clip(coordinate,self.x[0],self.x[-1]))
        error=np.asarray([np.interp(x,self.x,self.errors[:,i]) for i in range(2)])
        return np.asarray(self.curve(x)),float(error[0]),float(error[1])


def prepare_continuous_path(ik, target_at, initial_joints, *, samples, iterations):
    """Follow one arm branch through a dense tool route using scratch IK only."""
    coordinates=np.linspace(0.,1.,max(2,int(samples)))
    joints=[np.asarray(initial_joints,float).copy()];errors=[(0.,0.)]
    for coordinate in coordinates[1:]:
        position,rotation=target_at(float(coordinate))
        q,pe,re=bounded_continuation_solve(ik,position,rotation,joints[-1],iterations)
        joints.append(q);errors.append((pe,re))
    return ContinuousJointPath(coordinates,joints,errors)


def prepare_reviewed_free_path(ik,path,initial_joints,*,nominal_pivot,valve_offset,
        axis_world,angle_delta_rad,exiting,route_duration,control_dt,iterations,
        orientation_adjustment=None):
    """Shared G1/runtime free-route construction, entirely scratch geometry.

    path is the reviewed tool pose path (ReviewedPath). The caller supplies
    the actual start joint pose and handle angle difference from its reviewed
    endpoint. G1 can call this without claiming execution approval or creating
    a physical controller. Runtime uses exactly the same construction.
    """
    pivot=np.asarray(nominal_pivot,float);offset=np.asarray(valve_offset,float)
    axis=np.asarray(axis_world,float)
    def smooth(value):
        value=float(np.clip(value,0.,1.));return value*value*(3.-2.*value)
    def target_at(u):
        position,rotation,_=path.at(smooth(u))
        weight=smooth(min(1.,2.*(1.-u if exiting else u)))
        anchor=Rotation.from_rotvec(axis*angle_delta_rad*weight).as_matrix()
        position=pivot+anchor@(position-pivot)+weight*offset
        rotation=anchor@rotation
        if orientation_adjustment is not None:
            rotation=orientation_adjustment(rotation,weight)
        return position,rotation
    return prepare_continuous_path(ik,target_at,initial_joints,
        samples=max(2,int(np.ceil(route_duration/control_dt))+1),iterations=iterations)


def sample_reviewed_free_path(plan,elapsed,*,route_duration,settle_duration,natural):
    """Shared preview/runtime sampling with a continuous natural-down finish."""
    if settle_duration>0. and elapsed>=route_duration:
        u=float(np.clip((elapsed-route_duration)/settle_duration,0.,1.))
        u=u*u*(3.-2.*u)
        return (1.-u)*plan.joints[-1]+u*np.asarray(natural,float),0.,0.
    return plan.at(elapsed/route_duration)


class ReleasedCloseReturnPath:
    """Close-only joint return after the reviewed initial unhook movement.

    Keep the initial tool-route continuation exactly. Once the selected
    geometric release point has cleared the handle, stop following the old
    release wrist orientation. A single Hermite joint curve joins that actual
    arm pose and velocity to natural down, distributing the wrist return over
    the remaining time instead of accumulating it at an IK posture change.
    The caller must review collision geometry for the resulting route; this
    class makes no physical safety or success claim.
    """
    def __init__(self,reference,*,reference_duration,release_prefix_s,total_duration,natural):
        if not 0.<release_prefix_s<min(reference_duration,total_duration):
            raise ValueError('Close return needs an interior released join point')
        self.reference=reference
        self.reference_duration=float(reference_duration)
        self.release_prefix_s=float(release_prefix_s)
        self.total_duration=float(total_duration)
        self.natural=np.asarray(natural,float).copy()
        x=release_prefix_s/reference_duration
        self.release_joints=reference.at(x)[0]
        self.release_velocity=reference.curve.derivative()(x)/reference_duration
        self.active_join_s=self.release_prefix_s
        self.actual_connected=False
        self._make_bridge(self.release_joints,self.release_velocity)

    def _make_bridge(self,joints,velocity):
        self.return_duration=self.total_duration-self.active_join_s
        if self.return_duration<=0.:raise ValueError('Close return has no time left for continuous connection')
        self.bridge=CubicHermiteSpline([0.,self.return_duration],
            np.asarray([joints,self.natural]),np.asarray([velocity,np.zeros(7)]),axis=0)

    def connect_actual_release(self,joints,velocity,elapsed):
        """Runtime connection starts from observed joints, never recorded q."""
        self.active_join_s=float(elapsed)
        self.release_joints=np.asarray(joints,float).copy()
        self.release_velocity=np.asarray(velocity,float).copy()
        self._make_bridge(self.release_joints,self.release_velocity)
        self.actual_connected=True

    def at(self,elapsed):
        t=float(np.clip(elapsed,0.,self.total_duration))
        if t<self.active_join_s:
            return self.reference.at(t/self.reference_duration)
        return np.asarray(self.bridge(t-self.active_join_s)),0.,0.


def prepare_close_release_return(ik,path,initial_joints,*,policy,natural,
        nominal_pivot,valve_offset,axis_world,angle_delta_rad,control_dt,iterations,
        orientation_adjustment=None):
    """Shared G1/runtime constructor, opt-in for the reviewed close document."""
    if policy.get('version')!='close_released_joint_return_v1':
        raise ValueError('Unknown close release reconnection policy')
    reference_duration=float(policy['reference_tool_route_duration_s'])
    reference=prepare_reviewed_free_path(ik,path,initial_joints,
        nominal_pivot=nominal_pivot,valve_offset=valve_offset,axis_world=axis_world,
        angle_delta_rad=angle_delta_rad,exiting=True,route_duration=reference_duration,
        control_dt=control_dt,iterations=iterations,orientation_adjustment=orientation_adjustment)
    return ReleasedCloseReturnPath(reference,reference_duration=reference_duration,
        release_prefix_s=float(policy['release_prefix_s']),
        total_duration=float(policy['total_duration_s']),natural=natural)
