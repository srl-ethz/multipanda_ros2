import os
from math import asin, atan2, copysign, pi

import yaml


def _format_vector(values, expected_length, key_name):
    if not isinstance(values, (list, tuple)) or len(values) != expected_length:
        raise ValueError(f'Expected {expected_length} values for "{key_name}", got: {values}')
    return ' '.join(f'{float(value):.16g}' for value in values)


def _quaternion_xyzw_to_rpy(rotation_xyzw):
    if not isinstance(rotation_xyzw, (list, tuple)) or len(rotation_xyzw) != 4:
        raise ValueError(f'Expected 4 values for "rotation_xyzw", got: {rotation_xyzw}')
    x, y, z, w = [float(value) for value in rotation_xyzw]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = copysign(pi / 2.0, sinp)
    else:
        pitch = asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def _load_world_to_franka_base(hardware_layout_file):
    with open(os.path.expanduser(hardware_layout_file), 'r', encoding='utf-8') as file:
        layout_data = yaml.safe_load(file) or {}
    return layout_data['hardware_layout']['ros__parameters']['world_to_franka_base']


def _resolve_world_to_base_pose(world_to_franka_base, arm_id):
    if arm_id not in world_to_franka_base:
        raise KeyError(
            f'No world_to_franka_base entry for arm id "{arm_id}". '
            f'Available ids: {sorted(world_to_franka_base.keys())}'
        )

    transform = world_to_franka_base[arm_id]
    translation = transform['translation']
    rotation_xyzw = transform['rotation_xyzw']
    world_to_base_xyz = _format_vector(translation, 3, 'translation')
    world_to_base_rpy = _format_vector(_quaternion_xyzw_to_rpy(rotation_xyzw), 3, 'rpy')
    return world_to_base_xyz, world_to_base_rpy


def resolve_world_to_base_poses(hardware_layout_file, arm_ids):
    world_to_franka_base = _load_world_to_franka_base(hardware_layout_file)
    resolved = {}
    for arm_id in arm_ids:
        resolved[arm_id] = _resolve_world_to_base_pose(world_to_franka_base, arm_id)
    return resolved
