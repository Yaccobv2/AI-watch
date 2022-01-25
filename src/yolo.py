# pylint: disable=R1728
"""
Yolo ANN detector module. Using this module you can create any type of yolo network and
make predictions.
"""

import time
from typing import List, Tuple, Any

import cv2
import numpy as np


class Yolo:
    """
    Yolo ANN detector module. Using this module you can create any type of yolo network and
    make predictions.
    """
    def __init__(self, confidenceFilter: float = 0.5, threshold: float = 0.3,
                 yoloCfgFile: str = "./networks/yolo/cfg/yolov4-tiny.cfg",
                 yoloWeightsFile: str = "./networks/yolo/yolov4-tiny_best.weights",
                 yoloNamesFile: str = "./networks/yolo/cfg/yolov4.names"):
        """
        Init yolo ANN detector module.

        :param confidenceFilter: Value used to filter out low confidence detections
        :param threshold: Value used in non-maximum Suppression to filter out overlapping boundingboxes
        :param yoloCfgFile: path to file with yolo config
        :param yoloWeightsFile: path to file with network weights
        :param yoloNamesFile: path to file with classes names file

        """

        # system parameters
        self.confidence_filter = confidenceFilter
        self.threshold = threshold

        # files
        self.yolo_cfg_file = yoloCfgFile
        self.yolo_weights_file = yoloWeightsFile
        self.yolo_names_file = yoloNamesFile

        # network
        self.yolo = None
        self.outputlayers = None
        self.labels = None
        self.colors = None

    def init_network(self) -> None:
        """
        Initialize the yolo network.

        :return: None

        """
        # read neural network structure, weights and biases
        # noinspection PyBroadException
        try:
            self.yolo = cv2.dnn.readNetFromDarknet(self.yolo_cfg_file, self.yolo_weights_file)
            self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception as e:
            print("Can't load yolo config or weights: " + str(e))

        self.outputlayers = self.yolo.getUnconnectedOutLayersNames()

        # read labels
        # noinspection PyBroadException
        try:
            with open(self.yolo_names_file, 'r',encoding='UTF-8') as f:
                self.labels = f.read().splitlines()
        except Exception as e:
            print("Can't load yolo labels: " + str(e))

        # create rgb colors for every label
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3),
                                        dtype="uint8")

    def forward_pass(self, blob: List) -> Tuple[Any, float]:
        """
        Perform forward pass.

        :param blob: input frame formatted with cv2.dnn.blobFromImage function

        :return: The output of the network and time it took to perform forward pass

        """
        self.yolo.setInput(blob)

        # make forward pass and calculate its time
        start = time.time()
        layerOutputs = self.yolo.forward(self.outputlayers)
        end = time.time()

        runtime = end - start

        return layerOutputs, runtime

    def process_output(self, layerOutputs: Any, imageDimensions: Tuple):
        """
        Process output of the network.

        :param layerOutputs: output of yolo network from forward_pass function
        :param imageDimensions: tuple containing the size of the input frame

        :return: The list of dicts containing information about detected objects

        """
        outputs = []

        # initialize our lists of detected bounding boxes, confidences and class IDs for every grabbed frame
        boxes = []
        confidences = []
        classIDs = []

        # use output of ANN
        for output in layerOutputs:
            for detection in output:

                # calculate highest score and get it`s confidence number
                score = detection[5:]
                classId = np.argmax(score)
                confidence = score[classId]

                # if confidence is higher than selected value of CONFIDENCE_FILTER
                # create bounding box for every detection
                if confidence > self.confidence_filter:
                    box = detection[0:4] * np.array([imageDimensions[1], imageDimensions[0],
                                                     imageDimensions[1], imageDimensions[0]])
                    (centerX, centerY, width, height) = box.astype("int")

                    # get left corner coordinates of bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classId)

            # apply non-maxima suppression to overlapping bounding boxes with low confidence
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_filter,
                                    self.threshold)

            # check if any bounding box exists
            if len(idxs) > 0:

                # plot bounding boxes
                for i in idxs.flatten():
                    detectedObject = {
                        "x": boxes[i][0],
                        "y": boxes[i][1],
                        "w": boxes[i][2],
                        "h": boxes[i][3],
                        "color": tuple([int(c) for c in self.colors[classIDs[i]]]),
                        "label": self.labels[classIDs[i]],
                        "confidence": confidences[i]
                    }

                    outputs.append(detectedObject)

        outputs = [dict(tupleized) for tupleized in set(tuple(item.items()) for item in outputs)]

        return outputs
