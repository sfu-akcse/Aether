import cv2

def extract_z_coordinate(image, detection_result, z_value, base_value):
    if not detection_result.hand_landmarks:
        return None, base_value

    h, w, _ = image.shape
    hand_landmarks = detection_result.hand_landmarks[0]

    palm_indices = [0, 1, 5, 9, 13, 17]
    x_arr = []
    y_arr = []

    for i in palm_indices:
        x_arr.append(hand_landmarks[i].x * w)
        y_arr.append(hand_landmarks[i].y * h)

    padding = 20
    min_x = max(0, int(min(x_arr)) - padding)
    max_x = min(w, int(max(x_arr)) + padding)
    min_y = max(0, int(min(y_arr)) - padding)
    max_y = min(h, int(max(y_arr)) + padding)

    box_width = max_x - min_x
    box_height = max_y - min_y
    box_area = box_width * box_height

    if z_value == 0:
        base_value = box_area

    if base_value is None:
        return None, base_value

    z_offset = int(box_area**0.5 - base_value**0.5)
    return max(0, z_offset), base_value

def label_z_coordinate(image, detection_result, z_value, base_value):
    z_coordinate, base_value = extract_z_coordinate(image, detection_result, z_value, base_value)
    if not detection_result.hand_landmarks:
        return image, base_value

    h, w, _ = image.shape
    hand_landmarks = detection_result.hand_landmarks[0]

    palm_indices = [0, 1, 5, 9, 13, 17]
    x_arr = []
    y_arr = []

    # Calculate bounding box coordinates
    for i in palm_indices:
        x = hand_landmarks[i].x * w
        x_arr.append(x)
        y = hand_landmarks[i].y * h
        y_arr.append(y)
    
    padding = 20
    min_x = int(min(x_arr)) - padding
    max_x = int(max(x_arr)) + padding
    min_y = int(min(y_arr)) - padding
    max_y = int(max(y_arr)) + padding

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(w, max_x)
    max_y = min(h, max_y)

    box_width = max_x - min_x
    box_height = max_y - min_y
    box_area = box_width * box_height

    # Draw text and boxes
    if z_coordinate is None:
        cv2.putText(image, "Press 'r' to set Z=0", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else: 
        cv2.putText(image, f"Z: {z_coordinate}", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    return image, base_value
