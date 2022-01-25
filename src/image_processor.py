"""
Module responsible for whole program logic. It contains main loop and drawing functions.
"""

from typing import Tuple, List, Any, Union

import cv2

from phone_user_detector import detect_phone_users
from pose_detetion_module import PoseDetector
from yolo import Yolo


def start_detection(inputSource: Union[str, int] = 0, networkInputFrameSize: Tuple = (416, 416),
                    poseModelComplexity: int = 1) -> None:
    """
        Start program loop. You can break it using "q" key.

        :param inputSource: the video input source. By default
         it is your desktop camera int=0, but you can specify
          any file using string format f.e. "test-vide.mp4".
        :param networkInputFrameSize: the size of the input
         image specified in the yolo config file.
        :param poseModelComplexity: the value to specify the complexity of pose detection network.
         You can choose between 0, 1, 2, where 0 is the fastest and least accurate,
          2 is the slowest but has best accuracy

        :return: None

    """
    cap = cv2.VideoCapture(inputSource)

    # camera image size
    imageDimensions = (None, None)

    # create yolo
    yolo = Yolo()
    yolo.init_network()

    # create pose detector
    poseDetector = PoseDetector(model_complexity=poseModelComplexity)

    while True:

        # Capture frame-by-frame
        success, frame = cap.read()
        if success:
            # try:
            #     frame = cv2.resize(frame, (1280, 720))
            # except Exception as e:
            #     print("Can't resize the frame" + str(e))

            # if the frame dimensions are empty,
            # grab them
            if imageDimensions[0] is None\
                    or imageDimensions[1] is None:
                imageDimensions = frame.shape[:2]

            # create input matrix from frame, apply transformations
            # and pass it to the first layer of ANN
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, networkInputFrameSize, swapRB=True, crop=False)
            layerOutputs, runtime = yolo.forward_pass(blob=blob)
            print(runtime)

            results = yolo.process_output(layerOutputs=layerOutputs, imageDimensions=imageDimensions)
            detections = find_poses(poseDetector=poseDetector, frame=frame, detections=results,
                                    imageDimensions=imageDimensions)

            frame = draw_poses(detections=detections, frame=frame)

            results = detect_phone_users(detectionResults=results)

            frame = print_boundingboxes(results=results, frame=frame)

            cv2.imshow("frame", frame)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def print_boundingboxes(results: List[dict], frame: Any) -> Any:
    """
      Print detected boundingboxes and wrists.

      :param results: the list of dicts containing the detected objects
      :param frame: the captured frame

      :return: frame with printed data

    """
    for result in results:
        cv2.rectangle(frame, (result["x"], result["y"]),
                      (result["x"] + result["w"], result["y"] + result["h"]),
                      result["color"], 2)
        text = result["label"]+": "+ str(round(result["confidence"], 4))
        cv2.putText(frame, text, (result["x"], result["y"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, result["color"], 2)

        if "right_wrist" in result:
            cv2.circle(frame, (result["right_wrist"][0], result["right_wrist"][1]),
                       radius=5,
                       color=(235, 13, 18),
                       thickness=5)
        if "left_wrist" in result:
            cv2.circle(frame, (result["left_wrist"][0], result["left_wrist"][1]),
                       radius=5,
                       color=(235, 13, 18),
                       thickness=5)

    return frame


def find_poses(poseDetector: Any, frame: Any, detections: List[dict],
               imageDimensions: Tuple) -> Any:
    """
    Detect poses in given frame and return extended list
     of dicts with pose landmarks and images with drawn poses.

    :param imageDimensions: tuple containing the size of the input frame
    :param detections: the list of dicts containing the detected objects
    :param poseDetector: poseDetector object from pose_detection_module.py module
    :param frame: the captured frame

    :return: the list of dicts containing the detected objects with pose landmarks
     and images with drawn poses
    """

    for detection in filter(lambda detection: detection["label"] == "pedestrian", detections):
        detection = fix_coordinates(detection, imageDimensions)
        pedestrian = frame[detection["y"]:detection["y"] + detection["h"],
                     detection["x"]:detection["x"] + detection["w"]]
        detection["resultImg"] = poseDetector.find_pose(img=pedestrian)
        landmarks = poseDetector.get_pixel_positions(poseDetector.get_landmarks(), pedestrian)

        detection["landmarks"] = resize_to_original_shape(landmarks, detection["x"], detection["y"])
    return detections


def resize_to_original_shape(landmarks: List, x: int, y: int) -> List:
    """
    Resize the values from the pose detector to match the coordinates in the entire frame

    :param landmarks: detected body parts of the object
    :param x: x coordinate of left upper corner of boundingbox
    :param y: y coordinate of left upper corner of boundingbox

    :return: the list of the resized to original frame landmarks
    """
    for landmark in landmarks:
        landmark[1] += x
        landmark[2] += y
    return landmarks


def fix_coordinates(boundingbox: dict, imageDimensions: Tuple) -> dict:
    """
    Cut boundingbox dimensions if it is outside of the frame, to match the size of the frame

    :param boundingbox: the dict with the information about the detected object
    :param imageDimensions: tuple containing the size of the input frame

    :return: resized boundingbox
    """
    if boundingbox["x"] < 0:
        boundingbox["x"] = 0

    if boundingbox["y"] < 0:
        boundingbox["y"] = 0

    if boundingbox["x"] + boundingbox["w"] >= imageDimensions[1]:
        boundingbox["w"] = imageDimensions[1] - boundingbox["x"]

    if boundingbox["y"] + boundingbox["h"] >= imageDimensions[0]:
        boundingbox["h"] = imageDimensions[0] - boundingbox["y"]

    return boundingbox


def draw_poses(detections: List[dict], frame: Any) -> Any:
    """
    Draw detected poses.

    :param detections: the list of dicts containing the detected
     objects with the images from the find_poses function
    :param frame: the captured frame

    :return: the frame with drawn objects poses
    """
    for detection in detections:
        if "resultImg" in detection:
            frame[detection["y"]:detection["y"] + detection["h"],
            detection["x"]:detection["x"] + detection["w"]] = detection["resultImg"]

            del detection["resultImg"]

    return frame


if __name__ == "__main__":
    start_detection(networkInputFrameSize=(416, 416))
