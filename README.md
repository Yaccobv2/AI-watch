# AI-watch
AI system to detect pedestrians using cell phones.

#Purpose of the project

The aim of the project is to create a vision system capable of detecting whether a pedestrian is actively using a cell phone. The project is intended to increase pedestrian safety and reduce the number of accidents. The system is to detect pedestrians and the phones held in their hands, analyze the posture of each person and then decide whether the person is actively using the phone.


# Used tools
* darknet framework
* yolov4/yolov4-tiny
* google colabolatory
* python
* medipipe
* opencv2
* numpy
* pylint, pytest

# Dataset used to learn pedestrian and phones detector.

To learn pedestrian and cell phones detector I created my own dataset consisting of frames from videos available on youtube, which copyright allowed to use them freely.  The dataset consists of images of streets and sidewalks where pedestrians are present. 

Example image:
![212](https://user-images.githubusercontent.com/39679208/151044950-0bfdbe79-9c2a-487b-a697-612bd6be6a0b.png)

# Training yolov4-tiny
<img src="https://user-images.githubusercontent.com/39679208/151045240-82f8f4cb-9c70-4d17-9ead-79d0ad61581d.png" width="600" height="600" />

Yolov4 achieved a mAp of about 75%, but network speed on CUP did not allow for real-time detection.

# Video presentation
Phone detection by yolov4-tiny is not the best, but it allows you to work in real time. https://www.youtube.com/watch?v=dDOubt0FSmU
[![VIDEO](https://user-images.githubusercontent.com/39679208/151050032-398a99d3-2a34-4029-a078-49c88b7312ba.PNG)](https://youtu.be/dDOubt0FSmU)


# How to run this project
To run this project you must create a virtualenv with python, preferably version 3.9, and install the necessary libraries using: ```pip install requirements.txt```.
Now you can type python ```python main.py``` and run the program.


