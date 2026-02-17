#  Copyright (c) 2021 Franka Emika GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import os
from math import asin, atan2, copysign, pi

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node


def concatenate_ns(ns1, ns2, absolute=False):
    if(len(ns1) == 0):
        return ns2
    if(len(ns2) == 0):
        return ns1

    # check for /s at the end and start
    if(ns1[0] == '/'):
        ns1 = ns1[1:]
    if(ns1[-1] == '/'):
        ns1 = ns1[:-1]
    if(ns2[0] == '/'):
        ns2 = ns2[1:]
    if(ns2[-1] == '/'):
        ns2 = ns2[:-1]
    if(absolute):
        ns1 = '/' + ns1
    return ns1 + '/' + ns2


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


def _create_robot_state_publisher(
    context,
    *,
    franka_xacro_file,
    arm_id_1,
    arm_id_2,
    initial_positions_1,
    initial_positions_2,
    hardware_layout,
    load_gripper,
    ns,
):
    arm_id_1_value = arm_id_1.perform(context)
    arm_id_2_value = arm_id_2.perform(context)
    initial_positions_1_value = initial_positions_1.perform(context)
    initial_positions_2_value = initial_positions_2.perform(context)
    hardware_layout_file = os.path.expanduser(hardware_layout.perform(context))

    try:
        with open(hardware_layout_file, 'r', encoding='utf-8') as file:
            layout_data = yaml.safe_load(file) or {}
        world_to_franka_base = layout_data['hardware_layout']['ros__parameters']['world_to_franka_base']
        world_to_base_xyz_1, world_to_base_rpy_1 = _resolve_world_to_base_pose(
            world_to_franka_base,
            arm_id_1_value,
        )
        world_to_base_xyz_2, world_to_base_rpy_2 = _resolve_world_to_base_pose(
            world_to_franka_base,
            arm_id_2_value,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f'Failed to resolve world_to_franka_base for "{arm_id_1_value}" and "{arm_id_2_value}" '
            f'from "{hardware_layout_file}": {error}'
        ) from error

    robot_description = Command(
        [
            FindExecutable(name='xacro'),
            ' ',
            franka_xacro_file,
            ' arm_id_1:=',
            arm_id_1_value,
            ' arm_id_2:=',
            arm_id_2_value,
            ' hand_1:=',
            str(load_gripper).lower(),
            ' hand_2:=',
            str(load_gripper).lower(),
            ' initial_positions_1:=',
            initial_positions_1_value,
            ' initial_positions_2:=',
            initial_positions_2_value,
            f' world_to_base_xyz_1:="{world_to_base_xyz_1}"',
            f' world_to_base_rpy_1:="{world_to_base_rpy_1}"',
            f' world_to_base_xyz_2:="{world_to_base_xyz_2}"',
            f' world_to_base_rpy_2:="{world_to_base_rpy_2}"',
        ]
    )

    return [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            namespace=ns,
            parameters=[{'robot_description': robot_description}],
        )
    ]


def generate_launch_description():
    arm_id_1_param = "arm_id_1"
    arm_id_2_param = "arm_id_2"
    initial_positions_1_param = 'initial_positions_1'
    initial_positions_2_param = 'initial_positions_2'
    use_rviz_param = 'use_rviz'
    hardware_layout_param = 'hardware_layout'

    arm_id_1 = LaunchConfiguration(arm_id_1_param)
    arm_id_2 = LaunchConfiguration(arm_id_2_param)
    initial_positions_1 = LaunchConfiguration(initial_positions_1_param)
    initial_positions_2 = LaunchConfiguration(initial_positions_2_param)
    use_rviz = LaunchConfiguration(use_rviz_param)
    hardware_layout = LaunchConfiguration(hardware_layout_param)

    # Fixed variables
    load_gripper = True # We make gripper a fixed variable, mainly because parsing the argument 
                        # within generate_launch_description is a fairly unintuitive process, 
                        # and it's not worth doing just for a single boolean.
                        
    if(load_gripper): # mujoco scene file must be manually adjusted since there's no way to pass parameters
        scene_file = 'dual_scene.xml'
    else:
        scene_file = 'dual_scene_ng.xml'
    franka_xacro_file = os.path.join(get_package_share_directory('franka_description'), 'robots', 'sim',
                                     "dual_panda_arm_sim.urdf.xacro")
    xml_file = os.path.join(get_package_share_directory('franka_description'), 'mujoco', 'franka', scene_file)
    mjros_config_file = os.path.join(get_package_share_directory('franka_bringup'), 'config', 'sim',
                                     'dual_sim_controllers.yaml')
    hardware_layout_file = os.path.join(
        get_package_share_directory('franka_bringup'),
        'config',
        'hardware_layout.yaml')
    franka_bringup_path = get_package_share_directory('franka_bringup')
    ns=""

    # Joint state publisher setup
    jsp_source_list = [concatenate_ns(ns, 'joint_states', True)]
    if(load_gripper):
        jsp_source_list.append(concatenate_ns(ns, 'mj_left_gripper_sim_node/joint_states/joint_states', True))
        jsp_source_list.append(concatenate_ns(ns, 'mj_right_gripper_sim_node/joint_states/joint_states', True))

    node_joint_state_publisher = Node( # RVIZ dependency
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            namespace= ns,
            parameters=[
                {'source_list': jsp_source_list,
                 'rate': 30}],
    )

    # Others
    rviz_file = os.path.join(get_package_share_directory('franka_description'), 'rviz',
                             'visualize_dual_franka.rviz')
    

    return LaunchDescription([
        # Launch args
        DeclareLaunchArgument(
            use_rviz_param,
            default_value='false',
            description='Visualize the robot in Rviz'),
        DeclareLaunchArgument(
            arm_id_1_param,
            default_value='mj_left',
            description='Unique name of robot 1.'
        ),
        DeclareLaunchArgument(
            arm_id_2_param,
            default_value='mj_right',
            description='Unique name of robot 2.'
        ),
        DeclareLaunchArgument(
            initial_positions_1_param,
            default_value='"0.0 -0.785 0.0 -2.356 0.0 1.571 0.785"',
            description='Initial joint positions of robot 1. Must be enclosed in quotes, and in pure number.'
                        'Defaults to the "communication_test" pose.'),
        DeclareLaunchArgument(
            initial_positions_2_param,
            default_value='"0.0 -0.785 0.0 -2.356 0.0 1.571 0.785"',
            description='Initial joint positions of robot 2. Must be enclosed in quotes, and in pure number.'
                        'Defaults to the "communication_test" pose.'),
        DeclareLaunchArgument(
            hardware_layout_param,
            default_value=hardware_layout_file,
            description='Path to shared world-to-franka base transform parameters.'),

        Node(
            package='franka_robot_state_broadcaster',
            executable='hardware_layout_server',
            name='hardware_layout',
            parameters=[hardware_layout],
            output='screen',
        ),

        # Mujoco ros2 server launch
        IncludeLaunchDescription(
            FrontendLaunchDescriptionSource(franka_bringup_path + '/launch/sim/launch_mujoco_ros_server.launch'),
            launch_arguments={
                'use_sim_time': "true",
                'modelfile': xml_file,
                'verbose': "true",
                'ns': ns,
                'mujoco_plugin_config': mjros_config_file
                # 'mujoco_plugin_config': os.path.join(mjr2_control_path, 'example', 'ros2_control_plugins_example.yaml')

            }.items()
        ),

        # Miscellaneous
        OpaqueFunction(
            function=_create_robot_state_publisher,
            kwargs={
                'franka_xacro_file': franka_xacro_file,
                'arm_id_1': arm_id_1,
                'arm_id_2': arm_id_2,
                'initial_positions_1': initial_positions_1,
                'initial_positions_2': initial_positions_2,
                'hardware_layout': hardware_layout,
                'load_gripper': load_gripper,
                'ns': ns,
            },
        ),
        node_joint_state_publisher,

        Node( # RVIZ dependency
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '-c', concatenate_ns(ns, 'controller_manager', True)],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['franka_left_robot_state_broadcaster', '-c', concatenate_ns(ns, 'controller_manager', True)],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['franka_right_robot_state_broadcaster', '-c', concatenate_ns(ns, 'controller_manager', True)],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['multi_mode_controller', '-c', concatenate_ns(ns, 'controller_manager', True)],
            output='screen',
        ),
        Node(package='rviz2',
             executable='rviz2',
             name='rviz2',
             arguments=['--display-config', rviz_file],
             condition=IfCondition(use_rviz)
             )

    ])
