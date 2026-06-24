import cv2

def extract_xy_coordinates(image, detection_result):
    if not detection_result.hand_landmarks:
        return None

    h, w, _ = image.shape

    box_size = min(h, w)
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    hand_landmarks = detection_result.hand_landmarks[0]
    hand_indices = []

    for i in [0, 1, 2, 5, 9, 13, 17]:
        if 0.0 <= hand_landmarks[i].x <= 1.0 and 0.0 <= hand_landmarks[i].y <= 1.0:
            hand_indices.append(i)

    if not hand_indices:
        return None

    x_average = sum(hand_landmarks[i].x for i in hand_indices) / len(hand_indices)
    y_average = sum(hand_landmarks[i].y for i in hand_indices) / len(hand_indices)

    hand_x = int(x_average * w)
    hand_y = int(y_average * h)

    clamped_x = max(x1, min(hand_x, x2))
    clamped_y = max(y1, min(hand_y, y2))

    x = ((clamped_x - x1) / box_size) * 200 - 100
    y = 100 - ((clamped_y - y1) / box_size) * 200

    return {
        'pixel_x': hand_x,
        'pixel_y': hand_y,
        'x': round(x, 1),
        'y': round(y, 1),
    }

def draw_xy_coordinates(image, detection_result):
    coordinate_data = extract_xy_coordinates(image, detection_result)
    if coordinate_data is None:
        return image

    hand_x = coordinate_data['pixel_x']
    hand_y = coordinate_data['pixel_y']
    text = f"({coordinate_data['x']}, {coordinate_data['y']})"

    cv2.circle(image, (hand_x, hand_y), 8, (255, 0, 0), -1)
    cv2.putText(image, text, (hand_x + 12, hand_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return image
