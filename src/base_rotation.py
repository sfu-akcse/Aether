import cv2

def base_rotation_x(xy_coordinates, image):
    if xy_coordinates is None:
        return

    h, w = image.shape[:2]

    if xy_coordinates['x'] <= -850:
        cv2.arrowedLine(
            image,
            (100, h // 2),
            (20, h // 2),
            (0, 255, 0),
            5,
            tipLength=0.4
        )

    elif xy_coordinates['x'] >= 850:
        cv2.arrowedLine(
            image,
            (w - 100, h // 2),
            (w - 20, h // 2),
            (0, 255, 0),
            5,
            tipLength=0.4
        )
         
    