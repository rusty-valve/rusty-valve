"""Join the actual held TCP to the reviewed release reference.

Both saved TCPs are expressed in the handle frame. Their difference preserves
the current physical contact radius rather than returning to the reference
radius. The final orientation comes from the reviewed release pose, so actual
tracking tilt is not carried into retreat. This never drives the valve itself.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

VERSION = 'g1_release_pose_actual_start_v2'


def _rotation(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix), 1., atol=1e-6)):
        raise ValueError('G1 release needs a finite proper rotation matrix')
    return matrix


def _position(value):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError('G1 release needs a finite TCP position')
    return vector


class G1ReleaseReference:
    def __init__(self, end_pose, release_pose):
        end = end_pose['tcp_in_handle_frame']
        release = release_pose['tcp_in_handle_frame']
        self.delta_position = _position(release['position_m']) - _position(end['position_m'])
        self.delta_rotation = (_rotation(release['rotation_matrix']) @
                               _rotation(end['rotation_matrix']).T)
        self.release_rotation = _rotation(release['rotation_matrix']).copy()
        self.end_pose_id = end_pose.get('id')
        self.release_pose_id = release_pose.get('id')

    def goal_from_actual(self, start_position, start_rotation, handle_rotation):
        frame = _rotation(handle_rotation)
        _rotation(start_rotation)
        return (_position(start_position) + frame @ self.delta_position,
                frame @ self.release_rotation)

    def metadata(self):
        return dict(policy=VERSION, end_pose_id=self.end_pose_id,
            release_pose_id=self.release_pose_id,
            reference_frame='handle_at_actual_release_entry',
            reference_delta_position_m=self.delta_position.tolist(),
            reference_delta_rotation_matrix=self.delta_rotation.tolist(),
            reference_release_rotation_matrix=self.release_rotation.tolist(),
            orientation_target='reviewed_release_in_actual_handle_frame',
            reference_delta_rotation_deg=float(np.degrees(
                np.linalg.norm(Rotation.from_matrix(self.delta_rotation).as_rotvec()))),
            actual_start_required=True, current_contact_radius_preserved=True,
            score_changed=False)
