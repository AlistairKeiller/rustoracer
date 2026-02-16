# Install instructions
## Devcontainer
Rebuild and reopen in container
## Starting up scripts
```bash
cargo run --release --features ros
```
```bash
cd autodrive_f1tenth && colcon build && cd ..
source autodrive_f1tenth/install/setup.bash
ros2 launch autodrive_f1tenth simulator_bringup_headless.launch.py
```
```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```
```bash
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=autodrive_online_async.yaml
```
```bash
cd wall_follow && colcon build && cd ..
source wall_follow/install/setup.bash
ros2 launch wall_follow wall_follow.launch.py
```
```bash
cd disparity_extender && colcon build && cd ..
source disparity_extender/install/setup.bash
ros2 launch disparity_extender disparity_extender.launch.py
```