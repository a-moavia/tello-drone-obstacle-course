# tello-drone-obstacle-course
Group 06 - TIERS lab , University of Turku



Go to https://github.com/TIERS/drone_racing_ros2 and follow the instructions. 

After completing that, add the folder of "drone_racer" at ...\drone_racing_ros2_ws\src\

Open a three new terminals at drone_racing_ros2_ws

Terminal No. 01:

source /opt/ros/galactic/setup.bash

colcon build

source install/setup.bash

ros2 launch tello_driver teleop_launch.py



Terminal No. 02 

source /opt/ros/galactic/setup.bash

colcon build

source install/setup.bash

ros2 run drone_racer perception_node



Terminal No. 03

source /opt/ros/galactic/setup.bash

colcon build

source install/setup.bash

ros2 run drone_racer controller_sm




In case of any obnormality, please land the drone safely by using the following commands in a new terminal at drone_racing_ros2_ws


source /opt/ros/galactic/setup.bash

colcon build

source install/setup.bash

ros2 service call /drone1/tello_action tello_msgs/TelloAction '{cmd: 'land'}'

