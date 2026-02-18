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
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

_LAUNCH_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, _LAUNCH_ROOT)

from common.hardware_layout_utils import resolve_world_to_base_poses


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


def _create_description_nodes(
    context,
    *,
    franka_xacro_file,
    robot_ip_1,
    robot_ip_2,
    arm_id_1,
    arm_id_2,
    load_gripper_1,
    load_gripper_2,
    use_fake_hardware,
    fake_sensor_commands,
    hardware_layout,
    franka_controllers,
    ns,
):
    arm_id_1_value = arm_id_1.perform(context)
    arm_id_2_value = arm_id_2.perform(context)
    robot_ip_1_value = robot_ip_1.perform(context)
    robot_ip_2_value = robot_ip_2.perform(context)
    load_gripper_1_value = load_gripper_1.perform(context)
    load_gripper_2_value = load_gripper_2.perform(context)
    use_fake_hardware_value = use_fake_hardware.perform(context)
    fake_sensor_commands_value = fake_sensor_commands.perform(context)
    hardware_layout_file = hardware_layout.perform(context)

    try:
        world_to_base_poses = resolve_world_to_base_poses(
            hardware_layout_file,
            [arm_id_1_value, arm_id_2_value],
        )
        world_to_base_xyz_1, world_to_base_rpy_1 = world_to_base_poses[arm_id_1_value]
        world_to_base_xyz_2, world_to_base_rpy_2 = world_to_base_poses[arm_id_2_value]
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
            ' hand_1:=',
            load_gripper_1_value,
            ' hand_2:=',
            load_gripper_2_value,
            ' robot_ip_1:=',
            robot_ip_1_value,
            ' robot_ip_2:=',
            robot_ip_2_value,
            ' arm_id_1:=',
            arm_id_1_value,
            ' arm_id_2:=',
            arm_id_2_value,
            ' use_fake_hardware:=',
            use_fake_hardware_value,
            ' fake_sensor_commands:=',
            fake_sensor_commands_value,
            f' world_to_base_xyz_1:="{world_to_base_xyz_1}"',
            f' world_to_base_rpy_1:="{world_to_base_rpy_1}"',
            f' world_to_base_xyz_2:="{world_to_base_xyz_2}"',
            f' world_to_base_rpy_2:="{world_to_base_rpy_2}"',
        ]
    )

    return [
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[{'robot_description': robot_description}, franka_controllers],
            remappings=[('joint_states', 'franka/joint_states')],
            namespace=ns,
            output={
                'stdout': 'screen',
                'stderr': 'screen',
            },
            prefix=['stdbuf -o L'],
            on_exit=Shutdown(),
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=ns,
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
    ]


def generate_launch_description():
    robot_ip_1_parameter_name = 'robot_ip_1'
    robot_ip_2_parameter_name = 'robot_ip_2'
    
    load_gripper_1_parameter_name = 'load_gripper_1'
    load_gripper_2_parameter_name = 'load_gripper_2'

    arm_id_1_parameter_name = 'arm_id_1'
    arm_id_2_parameter_name = 'arm_id_2'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'
    use_rviz_parameter_name = 'use_rviz'
    hardware_layout_parameter_name = 'hardware_layout'

    robot_ip_1 = LaunchConfiguration(robot_ip_1_parameter_name)
    robot_ip_2 = LaunchConfiguration(robot_ip_2_parameter_name)

    arm_id_1 = LaunchConfiguration(arm_id_1_parameter_name)
    arm_id_2 = LaunchConfiguration(arm_id_2_parameter_name)

    load_gripper_1 = LaunchConfiguration(load_gripper_1_parameter_name)
    load_gripper_2 = LaunchConfiguration(load_gripper_2_parameter_name)
    
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)
    hardware_layout = LaunchConfiguration(hardware_layout_parameter_name)

    franka_xacro_file = os.path.join(get_package_share_directory('franka_description'), 'robots', 'real',
                                     'dual_panda_arm.urdf.xacro')
    rviz_file = os.path.join(get_package_share_directory('franka_description'), 'rviz',
                             'visualize_dual_franka.rviz')

    franka_controllers = PathJoinSubstitution(
        [
            FindPackageShare('franka_bringup'),
            'config', 'real',
            'dual_multimode.yaml',
        ]
    )
    hardware_layout_file = PathJoinSubstitution(
        [
            FindPackageShare('franka_bringup'),
            'config',
            'hardware_layout.yaml',
        ]
    )
    ns = 'paco'
    controller_manager_name = concatenate_ns(ns, 'controller_manager', True)
    robot_description_topic = concatenate_ns(ns, 'robot_description', True)
    joint_state_sources = [
        concatenate_ns(ns, 'franka/joint_states', True),
        concatenate_ns(ns, 'panda_gripper/joint_states', True),
    ]
    # rl_left_state_srv_name = concatenate_ns(ns, 'rl_left/get_robot_states', True)
    # rl_left_goal_topic = concatenate_ns(ns, 'rl_left/panda_joint_impedance_controller/desired_pose', True)
    # rl_right_state_srv_name = concatenate_ns(ns, 'rl_right/get_robot_states', True)
    # rl_right_goal_topic = concatenate_ns(ns, 'rl_right/panda_joint_impedance_controller/desired_pose', True)

    return LaunchDescription([
        DeclareLaunchArgument(
            robot_ip_1_parameter_name,
            description='Hostname or IP address of robot 1.'),
        DeclareLaunchArgument(
            robot_ip_2_parameter_name,
            description='Hostname or IP address of robot 2.'),
        DeclareLaunchArgument(
            arm_id_1_parameter_name,
            default_value="rl_left",
            description='Unique arm ID of robot 1.'),
        DeclareLaunchArgument(
            arm_id_2_parameter_name,
            default_value="rl_right",
            description='Unique arm ID of robot 2.'),
        DeclareLaunchArgument(
            use_rviz_parameter_name,
            default_value='true',
            description='Visualize the robot in Rviz'),
        DeclareLaunchArgument(
            use_fake_hardware_parameter_name,
            default_value='false',
            description='Use fake hardware'),
        DeclareLaunchArgument(
            fake_sensor_commands_parameter_name,
            default_value='false',
            description="Fake sensor commands. Only valid when '{}' is true".format(
                use_fake_hardware_parameter_name)),
        DeclareLaunchArgument(
            load_gripper_1_parameter_name,
            default_value='true',
            description='Use Franka Gripper as an end-effector, otherwise, robot 1 is loaded '
                        'without an end-effector.'),
        DeclareLaunchArgument(
            load_gripper_2_parameter_name,
            default_value='true',
            description='Use Franka Gripper as an end-effector, otherwise, robot 2 is loaded '
                        'without an end-effector.'),
        DeclareLaunchArgument(
            hardware_layout_parameter_name,
            default_value=hardware_layout_file,
            description='Path to shared world-to-franka base transform parameters.'),
        # Keep this global; the broadcaster currently resolves '/hardware_layout' explicitly.
        Node(
            package='franka_robot_state_broadcaster',
            executable='hardware_layout_server',
            name='hardware_layout',
            parameters=[hardware_layout],
            output='screen',
        ),
        OpaqueFunction(
            function=_create_description_nodes,
            kwargs={
                'franka_xacro_file': franka_xacro_file,
                'robot_ip_1': robot_ip_1,
                'robot_ip_2': robot_ip_2,
                'arm_id_1': arm_id_1,
                'arm_id_2': arm_id_2,
                'load_gripper_1': load_gripper_1,
                'load_gripper_2': load_gripper_2,
                'use_fake_hardware': use_fake_hardware,
                'fake_sensor_commands': fake_sensor_commands,
                'hardware_layout': hardware_layout,
                'franka_controllers': franka_controllers,
                'ns': ns,
            },
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            namespace=ns,
            parameters=[
                {'source_list': joint_state_sources,
                 'rate': 30}],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            namespace=ns,
            arguments=['joint_state_broadcaster', '-c', controller_manager_name],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            namespace=ns,
            arguments=['real_multi_mode_controller', '-c', controller_manager_name],
            output='screen',
            condition=UnlessCondition(use_fake_hardware),
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            namespace=ns,
            arguments=['rl_left_state_broadcaster', '-c', controller_manager_name],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            namespace=ns,
            arguments=['rl_right_state_broadcaster', '-c', controller_manager_name],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            namespace=ns,
            arguments=['rl_left_model_broadcaster', '-c', controller_manager_name],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            namespace=ns,
            arguments=['rl_right_model_broadcaster', '-c', controller_manager_name],
            output='screen',
        ),
        # Node(
        #     package="panda_motion_generators",
        #     executable="panda_poly_c2_joint_motion_generator_node",
        #     namespace=ns,
        #     arguments=["rl_left_joint_via_motion",
        #                rl_left_state_srv_name,
        #                "real_multi_mode_controller",
        #                "panda_joint_impedance_controller",
        #                rl_left_goal_topic]
        # ),
        # Node(
        #     package="panda_motion_generators",
        #     executable="panda_poly_c2_joint_motion_generator_node",
        #     namespace=ns,
        #     arguments=["rl_right_joint_via_motion",
        #                rl_right_state_srv_name,
        #                "real_multi_mode_controller",
        #                "panda_joint_impedance_controller",
        #                rl_right_goal_topic]
        # ),
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([PathJoinSubstitution(
        #         [FindPackageShare('franka_gripper'), 'launch', 'gripper.launch.py'])]),
        #     launch_arguments={robot_ip_1_parameter_name: robot_ip_1,
        #                       use_fake_hardware_parameter_name: use_fake_hardware}.items(),
        #     condition=IfCondition(load_gripper_1)

        # ),

        Node(package='rviz2',
             executable='rviz2',
             name='rviz2',
             arguments=['--display-config', rviz_file],
             remappings=[('/robot_description', robot_description_topic)],
             condition=IfCondition(use_rviz)
             )

    ])
