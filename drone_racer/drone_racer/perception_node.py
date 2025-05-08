import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.bridge = CvBridge()

        # Define QoS profile
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos
        )

        self.get_logger().info('Perception node started.')

        self.annotated_pub = self.create_publisher(Image, '/gate/image_annotated', 10)
        self.target_pub = self.create_publisher(Int32MultiArray, '/gate/target_coords', 10)
        self.stop_pub = self.create_publisher(Bool, '/stop_sign_detected', 10)

        # Initialize ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  

        self.get_logger().info('Perception with gate detection final started.')


    # this function detects fiducial markers, joins them to make a shape and finds the center of the shape to make it a target
    def detect_fiducial(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
        if ids is None or len(ids) == 0:
            return None

        centers = []
        for sq in corners:
            pts = sq[0]
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            centers.append((cx, cy))
            # Draw marker outline (orange)
            cv2.polylines(frame, [pts.astype(int)], True, (0, 128, 255), 2)
            # Draw bounding box around individual marker (blue)
            x, y, w, h = cv2.boundingRect(pts)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        pts = np.array(centers, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
        hull = cv2.convexHull(pts)
        M = cv2.moments(hull)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # Draw convex hull (green)
        cv2.polylines(frame, [hull], True, (0, 255, 0), 2)
        # Draw center (green)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        # Draw bounding box around convex hull (blue)
        x, y, w, h = cv2.boundingRect(hull)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return {'center': (cx, cy + 50), 'area': cv2.contourArea(hull), 'shape': 'fiducial'}


    # this is the general gate detection function

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect green gates ( that's the basic before it will go towards shape )
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Calculate the area of the gate found and it should be bigger than this to ensure no small objects are treated as gates. 
            if area > 8000:
                # Draw red bounding box around the entire green contour
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  

                # Extract the region of interest (ROI) around the contour
                roi = mask[y:y+h, x:x+w]
                roi_contours, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                # Find the largest shape (square, circle) in this ROI
                largest_shape_area = 0
                largest_shape_info = None

                for roi_cnt in roi_contours:
                    roi_area = cv2.contourArea(roi_cnt)
                    if roi_area > 100:  # Filter small noise contours
                        # Adjust contour coordinates to the original frame
                        roi_cnt = roi_cnt.copy()
                        roi_cnt[:, :, 0] += x
                        roi_cnt[:, :, 1] += y

                        # Approximate the contour to detect the shape
                        epsilon = 0.02 * cv2.arcLength(roi_cnt, True)
                        approx = cv2.approxPolyDP(roi_cnt, epsilon, True)

                        #now the two majors shapes were square and circle so the function tries to classify the shape before making its center as the target point

                        # Detect square (4 sides)
                        if len(approx) == 4:
                            x_shape, y_shape, w_shape, h_shape = cv2.boundingRect(approx)
                            aspect_ratio = float(w_shape) / h_shape
                            if 0.9 <= aspect_ratio <= 1.1:  # Check if it's roughly a square
                                if roi_area > largest_shape_area:
                                    largest_shape_area = roi_area
                                    cx, cy = x_shape + w_shape // 2, y_shape + h_shape // 2
                                    largest_shape_info = {
                                        'type': 'square',
                                        'x': x_shape,
                                        'y': y_shape,
                                        'w': w_shape,
                                        'h': h_shape,
                                        'cx': cx,
                                        'cy': cy,
                                        'area': roi_area
                                    }

                        # Detect circle
                        else:
                            (cx_circle, cy_circle), radius = cv2.minEnclosingCircle(roi_cnt)
                            circle_area = np.pi * (radius ** 2)
                            if 0.8 <= roi_area / circle_area <= 1.2:  # Check if contour is roughly circular
                                if roi_area > largest_shape_area:
                                    largest_shape_area = roi_area
                                    x_shape, y_shape = int(cx_circle - radius), int(cy_circle - radius)
                                    w_shape, h_shape = int(2 * radius), int(2 * radius)
                                    largest_shape_info = {
                                        'type': 'circle',
                                        'x': x_shape,
                                        'y': y_shape,
                                        'w': w_shape,
                                        'h': h_shape,
                                        'cx': int(cx_circle),
                                        'cy': int(cy_circle),
                                        'area': roi_area
                                    }

                # Process the largest shape if found
                if largest_shape_info:
                    shape_type = largest_shape_info['type']
                    x_shape = largest_shape_info['x']
                    y_shape = largest_shape_info['y']
                    w_shape = largest_shape_info['w']
                    h_shape = largest_shape_info['h']
                    cx = largest_shape_info['cx']
                    cy = largest_shape_info['cy']
                    shape_area = largest_shape_info['area']

                    # Update global target if this shape is the largest overall
                    if shape_area > max_area:
                        max_area = shape_area
                        target = (cx, cy)

                    # Draw purple bounding box around the largest shape, which is now our target

                    cv2.rectangle(frame, (x_shape, y_shape), (x_shape+w_shape, y_shape+h_shape), (128, 0, 128), 2)
                    cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)
                    cv2.putText(frame, shape_type.capitalize(), (x_shape, y_shape - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detect fiducial markers
        fiducial_result = self.detect_fiducial(frame)
        if fiducial_result and fiducial_result['area'] > max_area:
            target = fiducial_result['center']
            max_area = fiducial_result['area']

        # Publish target if found
        if target:
            msg_out = Int32MultiArray()
            msg_out.data = list(target)
            self.target_pub.publish(msg_out)

        # --- STOP SIGN DETECTION ---
        stop_detected = False

        # Define red color ranges for stop sign detection
        lower_red1 = np.array([0, 100, 80])
        upper_red1 = np.array([5, 255, 255])
        lower_red2 = np.array([175, 100, 80])
        upper_red2 = np.array([180, 255, 255])

        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:  # Minimum area threshold else it will detect every small red colour as stop sign and land
                # Check if the contour is approximately circular
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                circle_area = np.pi * (radius ** 2)
                circularity = area / circle_area

                # Approximate the contour to check if it has 8 sides (octagon)
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Consider it a stop sign if it's roughly circular or has approximately 8 sides
                if (0.9 <= circularity <= 1.1) or (len(approx) >= 7 and len(approx) <= 9):
                    stop_detected = True
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.putText(frame, "STOP SIGN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    break  # Process only the first valid stop sign

        # Publish stop sign detection result
        self.stop_pub.publish(Bool(data=stop_detected))

        # Publish annotated image
        self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))

        # Display the frame (for debugging)
        cv2.imshow("Camera", frame)
        cv2.imshow("Red Mask", red_mask)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()