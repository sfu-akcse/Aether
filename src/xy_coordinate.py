import cv2

def extract_xy_coordinates(image, detection_result):
    if not detection_result.hand_landmarks:
        return None

    h, w, _ = image.shape
    screen_center_x = w // 2
    screen_center_y = h // 2

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

    return {
        'pixel_x': hand_x,
        'pixel_y': hand_y,
        'x': hand_x - screen_center_x,
        'y': -(hand_y - screen_center_y),
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
