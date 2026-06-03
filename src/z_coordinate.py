import cv2

def label_z_coordinate(image, detection_result, z_value, base_value):
    # If no hands are detected, just return the image
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

    if z_value == 0:
        base_value = box_area

    # Draw text and boxes
    if base_value is None:
        cv2.putText(image, "Press 'r' to set Z=0", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else: 
        z_val = int(box_area**0.5 - base_value**0.5)
        cv2.putText(image, f"Z: {z_val}", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    return image, base_value