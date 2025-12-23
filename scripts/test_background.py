import freenect
import cv2
import numpy as np

from src.segment import get_background_room

def main():
    get_background_room()


if __name__ == "__main__":
    main()