"""
Phone users detection project.
"""
from src.image_processor import start_detection


def main() -> None:
    """
    Main
    :return: None
    """
    start_detection(networkInputFrameSize=(416, 416))


if __name__ == "__main__":
    main()
