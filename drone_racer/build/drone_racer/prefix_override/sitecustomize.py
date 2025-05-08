import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/sh0w0ff/drone_racing_ros2_ws/src/drone_racer/install/drone_racer'
