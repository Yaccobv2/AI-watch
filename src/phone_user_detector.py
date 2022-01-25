"""
Module responsible for detecting phone users using information from other modules.
"""

from math import sqrt
from typing import List, Dict, Tuple

import numpy as np

COLOR = np.random.randint(0, 255, 3, dtype="uint8")
LEFT_WRIST = 15
RIGHT_WRIST = 16
LANDMARK_VISIBILITY = 0.5


def detect_phone_users(detectionResults: List[Dict]) -> List[Dict]:
    """
    Check if there are any phones users based on given information.

    :param detectionResults: the list of dicts containing information about detected objects with pose landmarks

    :return: the list of dicts with updated information about detected objects

    """
    if len(detectionResults) != 0:
        pedestrians, phones = split_detections(detectionResults=detectionResults)
        pedestrians = check_if_phone_insight(pedestrians=pedestrians, phones=phones)
        pedestrians = check_wrists(pedestrians=pedestrians)
        return pedestrians + phones

    return detectionResults


def split_detections(detectionResults: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Split detected phones and pedestrians into diffrent lists

    :param detectionResults: the list of dicts containing information about detected objects with pose landmarks

    :return: two lists containing pedestrians and phones.

    """
    pedestrians = []
    phones = []

    for detectionResult in detectionResults:
        if detectionResult["label"] == "phone":
            phones.append(detectionResult)
        if detectionResult["label"] == "pedestrian":
            pedestrians.append(detectionResult)

    return pedestrians, phones


def check_if_phone_insight(pedestrians: List, phones: List) -> List[Dict]:
    """
    Check if any pedestrian boundingbox contains the phone boundingbox insight

    :param pedestrians: the list of information about the detected objects with label pedestrian
    :param phones: the list of information about detected objects with label phone

    :return: updated list with information if it contains phone insight

    """
    for pedestrian in pedestrians:
        pedestrian["isPhone"] = False
        for phone in phones:
            if find_point(pedestrian["x"], pedestrian["y"], pedestrian["w"], pedestrian["h"],
                          [(phone["x"], phone["y"]), (phone["x"] + phone["w"], phone["y"] + +phone["h"])]):
                pedestrian["isPhone"] = True

    return pedestrians


def find_point(x: int, y: int, w: int, h: int, corners: List[Tuple]) -> bool:
    """
     Check if the object with given corners is insight the boundingbox with coordinates x,y and dimensions w,h

     :param x: x coordinate of the boundingbox
     :param y: y coordinate of the boundingbox
     :param w: width of the boundingbox
     :param h: height of the boundingbox
     :param corners: the list of tuples containing left upper corner and right lower corner of boundingbox of
      detected object

     :return: True if object is insight, False if not

     """
    for corner in corners:
        if x < corner[0] < x + w and y < corner[1] < y + h:
            return True

    return False


def check_wrists(pedestrians: List[Dict]) -> List[Dict]:
    """
       Check if the object with given corners is insight the boundingbox with coordinates x,y and dimensions w,h

       :param pedestrians:

       :return: True if object is insight, False if not

    """
    for pedestrian in pedestrians:
        if pedestrian["isPhone"]:
            if is_upper_body_detected(pedestrian):
                # noinspection PyBroadException
                try:
                    pedestrian["left_wrist"] = (pedestrian["landmarks"][LEFT_WRIST][1],
                                                pedestrian["landmarks"][LEFT_WRIST][2])
                    pedestrian["right_wrist"] = (pedestrian["landmarks"][RIGHT_WRIST][1],
                                                 pedestrian["landmarks"][RIGHT_WRIST][2])

                    dist = calculate_distance((pedestrian["landmarks"][LEFT_WRIST][1],
                                               pedestrian["landmarks"][LEFT_WRIST][2]),
                                              (pedestrian["landmarks"][7][1], pedestrian["landmarks"][7][2]))

                    compDepth = compensate_depth(dist, pedestrian["w"])
                    print("Distance: " + str(dist) + ", width: " + str(pedestrian["w"]) + "Compensated depth: " +
                          str(compDepth)
                          )
                    if compDepth < 100:
                        pedestrian["label"] = "PHONE USER!!!!!!!"
                        pedestrian["color"] = (255, 255, 255)
                except Exception as e:
                    print("Wrist is not detected in this iteration: " + str(e))

    return pedestrians


def is_upper_body_detected(pedestrian: Dict) -> bool:
    """
    Check upper body of the pedestrian was detected

    :param pedestrian: the dict containing information about detected pedestrian

    :return: True if upper body was detected, False if not

    """
    for landmark in pedestrian["landmarks"][:24]:
        if landmark[4] < LANDMARK_VISIBILITY:
            return False

    return True


def calculate_distance(point1: Tuple, point2: Tuple) -> float:
    """
    Calculate distance between two points

    :param point1: tuple containing x and y coordinates of point 1
    :param point2: tuple containing x and y coordinates of point 2

    :return: float value of the distance between the points
    """
    return sqrt(pow((point2[0] - point1[0]), 2) + pow((point2[1] - point1[1]), 2))


def compensate_depth(distance: float, bbWidth: int) -> float:
    """
    Compensate depth to get accurate results when object is changing his distance from the camera

    :param distance: distance between left wrist and left ear
    :param bbWidth: width of the boundingox

    :return: float value of compensated width between the wrist and the shoulder
    """
    return distance - 0.5086 * bbWidth + 13.048
