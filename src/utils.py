import cv2

def draw_xy_coordinates(image, detection_result):
    if not detection_result.hand_landmarks:
        return image
    
    # define the center of the image (screen)
    h, w, _ = image.shape
    screen_center_x = w // 2
    screen_center_y = h // 2

    # print the dot on the center, 
    for hand_landmarks in detection_result.hand_landmarks:
        hand_indices = []

        for i in  [0, 1, 2, 5, 9, 13, 17]:
            if (
                0.0 <= hand_landmarks[i].x <= 1.0 
                and
                0.0 <= hand_landmarks[i].y <= 1.0 
            ):
                hand_indices.append(i)

        if not hand_indices:
            continue

        # hand_x 
        x_sum = 0
        for i in hand_indices:
            x_sum = x_sum + hand_landmarks[i].x  

        x_average = x_sum / len(hand_indices)   
        hand_x = int(x_average * w)            

        # hand_y 
        y_sum = 0
        for i in hand_indices:
            y_sum = y_sum + hand_landmarks[i].y  
            
        y_average = y_sum / len(hand_indices)   
        hand_y = int(y_average * h) 

        cv2.circle(image, (hand_x, hand_y), 8, (255, 0, 0), -1) # blue 

        # calculating the coordinates of the hand based on the center
        final_coordinate_x = hand_x - screen_center_x
        final_coordinate_y = -(hand_y - screen_center_y)

        # printing coordinates next to hand center
        text = f'({final_coordinate_x}, {final_coordinate_y})'
        cv2.putText(image, text, (hand_x + 12, hand_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return image