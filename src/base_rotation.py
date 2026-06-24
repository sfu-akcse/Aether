import cv2

def base_rotation_x(xy_coordinates, image):
    if xy_coordinates is None:
        return

    h, w = image.shape[:2]

    box_size = min(h, w)
    x1 = (w - box_size) // 2
    x2 = x1 + box_size

    left_center_x = x1 // 2
    right_center_x = (x2 + w) // 2

    if xy_coordinates['x'] <= -100:
        cv2.arrowedLine(
            image,
            (left_center_x + 40, h // 2),
            (left_center_x - 40, h // 2),
            (0, 255, 0),
            5,
            tipLength=0.4
        )

    elif xy_coordinates['x'] >= 100:
        cv2.arrowedLine(
            image,
            (right_center_x - 40, h // 2),
            (right_center_x + 40, h // 2),
            (0, 255, 0),
            5,
            tipLength=0.4
        )

def border_box(image, alpha=0.3):
    h, w, c = image.shape

    box_size = min(h, w)
    x1 = w // 2 - box_size // 2
    x2 = w // 2 + box_size // 2

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (x1, h), (0, 0, 255), -1)
    cv2.rectangle(overlay, (x2, 0), (w, h), (0, 0, 255), -1)

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    