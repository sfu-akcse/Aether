import launch
import launch_ros.actions


def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='py_pubsub',
            executable='talker',
            name='talker'),

        launch_ros.actions.Node(
            package='py_pubsub',
            executable='listener',
            name='listener')
    ])