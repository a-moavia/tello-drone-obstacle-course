import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Bool
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time

from tello_msgs.srv import TelloAction

class GateRaceController(Node):
    def __init__(self):
        super().__init__('gate_race_controller')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_target = self.create_subscription(Int32MultiArray, '/gate/target_coords', self.target_callback, qos)
        self.sub_stop = self.create_subscription(Bool, '/stop_sign_detected', self.stop_callback, qos)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mover=0
        self.state = 'TAKEOFF'
        self.alignment_phase = 'HORIZONTAL'  # Sub-state for ALIGNING: 'HORIZONTAL' or 'VERTICAL'
        self.target = None
        self.last_seen_time = 0
        self.recovery_start = None

        # Camera frame dimensions
        self.IMAGE_WIDTH = 960
        self.IMAGE_HEIGHT = 720  # Assuming a 640x480 resolution

        # Tolerances for alignment (in pixels)
   
        self.CENTER_TOLERANCE_X = 40 # Horizontal tolerance
        self.CENTER_TOLERANCE_Y = 5  # Vertical tolerance (y-axis)

        # Gains for proportional control
        self.HORIZONTAL_GAIN = 1.0  # For angular.z (yaw)
        self.VERTICAL_GAIN = 1.0   # For linear.z (up/down)

        # Scaling factors to adjust alignment speed (values between 0 and 1)

        self.HORIZONTAL_SCALING_FACTOR = 0.001 # Scales the horizontal alignment speed
        self.VERTICAL_SCALING_FACTOR = 0.003  # Scales the vertical alignment speed

        self.stop_flag = False
        self.forward_to_stop_start = None
        self.land_sent = False
        self.takeoff_start= time.time()
        self.tello_client = self.create_client(TelloAction, '/tello_action')

        # Wait for service to be ready

        while not self.tello_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for /tello_action FINAL service...')

        # Send takeoff command

        takeoff_req = TelloAction.Request()
        takeoff_req.cmd = 'takeoff'
        self.tello_client.call_async(takeoff_req)
        self.get_logger().info('🚁 Takeoff command sent.')

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('🚦 GateRaceController Final initialized.')

    def target_callback(self, msg):
        self.target = (msg.data[0], msg.data[1])
        self.last_seen_time = time.time()

    def stop_callback(self, msg):
        if msg.data and self.state not in ['APPROACHING_STOP', 'LANDING'] and not self.target:
            self.state = 'APPROACHING_STOP'
            self.forward_to_stop_start = time.time()
            self.get_logger().info('🛑 STOP sign detected → APPROACHING_STOP.')

    def control_loop(self):
        cmd = Twist()
        now = time.time()

        if self.state == 'TAKEOFF':
            #It takes off and remains stationary for 15 seconds to get stable
            if now - self.takeoff_start > 15.0:
                self.state = 'ASCENDING'
                self.ascend_start = now
                self.get_logger().info('⬆️ Starting ascent to higher altitude.')

        elif self.state == 'ASCENDING':
            # Send upward velocity to ascend more in order to get to the gate initial positions
            cmd.linear.z = 0.3
            if now - self.ascend_start > 2.5:
                cmd.linear.z = 0.0
                self.state = 'SEARCHING'
                self.get_logger().info('🔁 Reached desired altitude → SEARCHING.')
            

        if self.state == 'APPROACHING_STOP':
            # Move towards the stop sign after recognizing it to land in the designated area
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
            if now - self.forward_to_stop_start > 2.25:
                self.state = 'LANDING'
                self.get_logger().info('🛬 Reached STOP sign → LANDING.')

        elif self.state == 'LANDING':
            # Safe landing of the drone in the designated spot
            if not self.land_sent:
                self.land_sent = True
                land_req = TelloAction.Request()
                land_req.cmd = 'land'
                self.tello_client.call_async(land_req)
                self.get_logger().info('🛬 Land command sent.')
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        elif self.state == 'SEARCHING':
            # Searching the gate in drone's view by continous rightward rotation
            cmd.angular.z = -0.1
            if self.target and now - self.last_seen_time < 2.0:
                self.state = 'ALIGNING'
                self.alignment_phase = 'HORIZONTAL'  # Start with horizontal alignment
                self.get_logger().info('🎯 Gate found → ALIGNING (Horizontal phase)')

        elif self.state == 'ALIGNING':
            # Alligning towards the middle of the gate after gate recognition
            x, y = self.target
            # Calculate errors for both axes
            error_x = x - self.IMAGE_WIDTH // 2
            error_y = y - self.IMAGE_HEIGHT // 2

            if self.alignment_phase == 'HORIZONTAL':
                # First, align horizontally (x-axis) within the tolerance 
                if abs(error_x) < self.CENTER_TOLERANCE_X:
                    #if within tolerance move to the next phase
                    self.alignment_phase = 'VERTICAL'
                    self.get_logger().info('✅ Horizontally aligned → ALIGNING (Vertical phase)')
                else:
                    # if it is not within tolerance, it should re allign
                    cmd.angular.z = -self.HORIZONTAL_GAIN * self.HORIZONTAL_SCALING_FACTOR * error_x  # Adjust scaled yaw
                    cmd.linear.z = 0.0  # Ensure no vertical movement during horizontal alignment
                    self.get_logger().debug(f'Horizontal alignment: x-error={error_x}, cmd.angular.z={cmd.angular.z}')

            elif self.alignment_phase == 'VERTICAL':
                # Then, align vertically (y-axis) using up/down movement within tolerance
                if error_y < self.CENTER_TOLERANCE_Y:
                    #if within the tolerance, move to the next phase
                    self.state = 'MOVING'
                    self.forward_start = now
                    self.get_logger().info('✅ Vertically aligned → MOVING through gate')
                else:
                    # if it is not within tolerance, it should re allign
                    cmd.angular.z = 0.0  # Ensure no horizontal movement during vertical alignment
                    cmd.linear.z = -self.VERTICAL_GAIN * self.VERTICAL_SCALING_FACTOR * error_y  # Adjust height with scaled gain
                    self.get_logger().debug(f'Vertical alignment: y-error={error_y}, cmd.linear.z={cmd.linear.z}')

        elif self.state == 'MOVING':
            # State to just move the drone towards and through the gate, can have different initial and later velocity, if needed
            if self.mover == 0:
                cmd.linear.x = 0.35
                self.mover = 1
            else:
                cmd.linear.x = 0.35

            if now - self.forward_start > 4.0:
                self.state = 'RECOVERY'
                self.recovery_start = now
                self.get_logger().info('⏭️ Passed gate → RECOVERY.')

        elif self.state == 'RECOVERY':
            # state to recover and search for new gates after passing through a gate
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            if now - self.recovery_start > 2.0:
                self.state = 'SEARCHING'
                self.target = None
                self.get_logger().info('🔁 Back to SEARCHING.')

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = GateRaceController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()