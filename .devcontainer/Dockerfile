FROM docker.io/ros:jazzy-ros-core

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN apt-get update && apt-get install -y gcc libclang-dev git ros-$ROS_DISTRO-ackermann-msgs ros-$ROS_DISTRO-tf2-msgs ros-$ROS_DISTRO-foxglove-bridge ros-$ROS_DISTRO-slam-toolbox python3-colcon-common-extensions
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
