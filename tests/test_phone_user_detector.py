# pylint: skip-file
"""Test phone_user_detector.py"""

from src.phone_user_detector import check_if_phone_insight


def test_check_if_phone_insight():
    pedestrians =[{'x': 147, 'y': 121, 'w': 156, 'h': 359, 'color': (102, 220, 225), 'label': 'pedestrian',
      'confidence': 0.9966046214103699, 'isPhone': False}]
    phones = [{'x': 212, 'y': 224, 'w': 36, 'h': 44, 'color': (95, 179, 61),
               'label': 'phone', 'confidence': 0.827703058719635}]
    assert check_if_phone_insight(pedestrians=pedestrians, phones=phones)[0]["isPhone"] is True

