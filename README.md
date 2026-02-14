# Install instructions
## Devcontainer
Rebuild and reopen in container
## Starting up scripts
```bash
cargo run --release --features ros
```
```bash
cd autodrive_f1tenth && colcon build && cd ..
source autodrive_f1tenth/install/setup.bash && source /opt/ros/$ROS_DISTRO/setup.bash
ros2 launch autodrive_f1tenth simulator_bringup_headless.launch.py
```
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=autodrive_online_async.yaml
```
```bash
cd wall_follow && colcon build && cd ..
source wall_follow/install/setup.bash && source /opt/ros/$ROS_DISTRO/setup.bash
ros2 launch wall_follow wall_follow.launch.py
```