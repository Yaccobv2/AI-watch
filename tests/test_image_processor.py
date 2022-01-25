# pylint: skip-file
"""Test image_processor.py"""

from src.image_processor import resize_to_original_shape, fix_coordinates


def test_resize_to_original_shape():
    landmarks = [[0, 82, 23, -1.085, 0.9996267557144165], [1, 87, 16, -1.0136, 0.9993909597396851]]
    result_landmarks = [[0, 243, 244, -1.085, 0.9996267557144165], [1, 248, 237, -1.0136, 0.9993909597396851]]
    x = 161
    y = 221
    results = resize_to_original_shape(landmarks=landmarks, x=x, y=y)
    assert results[0][1] is result_landmarks[0][1] and results[0][2] is result_landmarks[0][2] \
           and results[1][1] is result_landmarks[1][1] and results[1][2] is result_landmarks[1][2]


def test_fix_coordinates():
    bbox = {'x': -10, 'y': 150, 'w': 177, 'h': 500, 'color': (102, 220, 225),
            'label': 'pedestrian', 'confidence': 0.6499353051185608}
    correctBbox = {'x': 0, 'y': 150, 'w': 177, 'h': 450,
                   'color': (102, 220, 225), 'label': 'pedestrian', 'confidence': 0.6499353051185608}
    results = fix_coordinates(boundingbox=bbox, imageDimensions=(600, 400))
    assert results['x'] == correctBbox['x'] and results['y'] == correctBbox['y'] and \
           results['w'] == correctBbox['w'] and results['h'] == correctBbox['h']
