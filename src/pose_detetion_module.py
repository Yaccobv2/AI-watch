# pylint: skip-file
"""
Pose detection module created with mediapipe
"""
import math

import cv2
import mediapipe as mp


class PoseDetector:
    """
    Pose detection class.
    """

    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """Initializes PoseDetector object.

            Args:
              static_image_mode: Whether to treat the input images as a batch of static
                and possibly unrelated images, or a video stream. See details in
                https://solutions.mediapipe.dev/pose#static_image_mode.
              model_complexity: Complexity of the pose landmark model: 0, 1 or 2. See
                details in https://solutions.mediapipe.dev/pose#model_complexity.
              smooth_landmarks: Whether to filter landmarks across different input
                images to reduce jitter. See details in
                https://solutions.mediapipe.dev/pose#smooth_landmarks.
              enable_segmentation: Whether to predict segmentation mask. See details in
                https://solutions.mediapipe.dev/pose#enable_segmentation.
              smooth_segmentation: Whether to filter segmentation across different input
                images to reduce jitter. See details in
                https://solutions.mediapipe.dev/pose#smooth_segmentation.
              min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for person
                detection to be considered successful. See details in
                https://solutions.mediapipe.dev/pose#min_detection_confidence.
              min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
                pose landmarks to be considered tracked successfully. See details in
                https://solutions.mediapipe.dev/pose#min_tracking_confidence.
            """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.static_image_mode, self.model_complexity,
                                      self.smooth_landmarks, self.enable_segmentation,
                                      self.smooth_segmentation, self.min_detection_confidence,
                                      self.min_tracking_confidence)

        self.results = None
        self.img = None

    def find_pose(self, img, draw=True):
        """Find joint detection

            Args:
                img: frame to process
                draw: Whether to draw detected points on given frame
                draw_3d_plot: Whether to draw 3d plot

            return:
                img: processed frame
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if draw:
            if self.results.pose_landmarks:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mp_pose.POSE_CONNECTIONS)

        self.img = img
        return img

    def get_connections(self):
        """Get connections list

            return:
                POSE_CONNECTIONS: connections list
        """
        return self.mp_pose.POSE_CONNECTIONS

    def get_world_landmarks(self):
        """Find joint detection

            return:
                pose_world_landmarks: landmarks
        """
        return self.results.pose_world_landmarks

    def get_landmarks(self):
        """Find joint detection

            return:
                pose_world_landmarks: landmarks
        """
        return self.results.pose_landmarks

    @staticmethod
    def get_pixel_positions(landmarks, img):
        """Find joint detection

            Args:
                landmarks: detected body parts
                img: captured frame

            return:
                lm_list: list of landmarks with values in pixels
        """
        lm_list = []
        if landmarks:
            for lm_id, l_m in enumerate(landmarks.landmark):
                w_h, w_w, w_c = img.shape
                p_x, p_y, p_z = int(l_m.x * w_w), int(l_m.y * w_h), round(l_m.z, 4)
                lm_list.append([lm_id, p_x, p_y, p_z, l_m.visibility])

        return lm_list

    def get_landmarks_list(self):
        """Find joint detection
               Args:

               return:
                   list of detected landmarks with their coordinates
         """

        list_of_landmarks = []

        if self.results.pose_landmarks is not None:
            for landmark_id, landmark in enumerate(self.results.pose_landmarks.landmark):
                list_of_landmarks.append([landmark_id, landmark.x, landmark.y, landmark.z])

        return list_of_landmarks

    def get_world_landmarks_list(self):
        """Find joint detection
               Args:

               return:
                   list of detected world landmarks with their coordinates
         """

        list_of_landmarks = []

        if self.results.pose_world_landmarks is not None:
            for landmark_id, landmark in enumerate(self.results.pose_world_landmarks.landmark):
                list_of_landmarks.append([landmark_id, landmark.x, landmark.y, landmark.z, landmark.visibility])

        return list_of_landmarks



    def get_angle(self, img, lm_list, set_of_joints):
        """Find joint detection
                Args:
                    img: frame to process
                    set_of_joints: set of 3 points that define joint

                return:
                    angle in degrees
        """
        coords = []
        for joint_id in set_of_joints:
            for landmark in lm_list:
                if landmark[0] == joint_id:
                    coords.append(landmark)

        if len(coords) == 3:
            angle = self.measure_angle(coords)
            print(angle)
            cv2.putText(img, str(int(angle)), (coords[1][1], coords[1][2] - 35),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            return angle, coords

        return None

    @staticmethod
    def measure_angle(joints_ladmarks):
        """Find joint detection
                Args:
                    joints_ladmarks: 3 points between which the angle will be measured

                return:
                    angle in degrees
        """
        a_vector = [joints_ladmarks[1][2] - joints_ladmarks[0][2], joints_ladmarks[1][1] - joints_ladmarks[0][1]]
        # joints_ladmarks[1][3]-joints_ladmarks[0][3]]

        b_vector = [joints_ladmarks[1][2] - joints_ladmarks[2][2], joints_ladmarks[1][1] - joints_ladmarks[2][1]]
        # joints_ladmarks[1][3]-joints_ladmarks[2][3]]

        # dot_product = (a_vector[0] * b_vector[0]) + (a_vector[1] * b_vector[1]) + (a_vector[2] * b_vector[2])

        dot_product = (a_vector[0] * b_vector[0]) + (a_vector[1] * b_vector[1])

        a_length = math.sqrt(a_vector[0] ** 2 + a_vector[1] ** 2)
        b_length = math.sqrt(b_vector[0] ** 2 + b_vector[1] ** 2)

        if (a_length * b_length) != 0:
            return round(math.degrees(math.acos(dot_product / (a_length * b_length))), 2)

        return None

    def get_all_angles(self, img, lm_list, sets_of_joints):
        """Find joint detection
                Args:
                    img: frame to process
                    lm_list: list of all points found by neural network
                    set_of_joints: list of all joints (defined by 3 points)
                                    where the angle will be measured

                return:
                    ret: angles and coordinates of all joints in set_of_joints
        """
        ret = []
        for set_of_joints in sets_of_joints:
            temp_angle, temp_coords = self.get_angle(img, lm_list, set_of_joints)
            ret.append([temp_angle, temp_coords])

        return ret
