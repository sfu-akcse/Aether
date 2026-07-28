import os
import time
import glob
import json
import cv2
import numpy as np
from base_rotation import base_rotation_x


def render_manual_mode(image, input_coordinates, up_down, wrist_turn, grab):
    """Handles rendering and JSON emission for manual mode."""
    h, w, _ = image.shape
    box_size = min(h, w)
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    hand_x = input_coordinates["x"]
    hand_y = input_coordinates["y"]

    clamped_x = max(x1, min(hand_x, x2))
    clamped_y = max(y1, min(hand_y, y2))

    x_value = ((clamped_x - x1) / box_size) * 200 - 100
    y_value = 100 - ((clamped_y - y1) / box_size) * 200

    manual_xyz = {
        "pixel_x": hand_x,
        "pixel_y": hand_y,
        "x": round(x_value, 1),
        "y": round(y_value, 1),
        "z": input_coordinates["z"]
    }

    overlay = image.copy()
    box_radius = max(75, 75 + int(input_coordinates["z"]))

    center_x = int(hand_x)
    center_y = int(hand_y)

    left = center_x - box_radius
    right = center_x + box_radius
    top = center_y - box_radius
    bottom = center_y + box_radius

    inactive_colour = (180, 180, 180)
    active_colour_wr = (255, 255, 0)
    active_colour_ud = (0, 255, 255)

    box_colour = (
        active_colour_ud if up_down == "center"
        else inactive_colour
    )

    circle_colour = (
        active_colour_wr if wrist_turn == "center"
        else inactive_colour
    )

    # Draw the transparent box
    cv2.rectangle(
        overlay,
        (left, top),
        (right, bottom),
        box_colour,
        -1
    )

    circle_radius = max(15, box_radius // 4)

    cv2.circle(
        overlay,
        (center_x, center_y),
        circle_radius,
        circle_colour,
        -1                  
    )

    triangle_size = 25
    if grab == False:  
        triangle_gap = 75
    elif grab == True:
        triangle_gap = 10

    # Up triangle
    up_triangle = np.array([
        [center_x, top - triangle_gap - triangle_size],
        [center_x - triangle_size, top - triangle_gap],
        [center_x + triangle_size, top - triangle_gap]
    ], dtype=np.int32)

    # Down triangle
    down_triangle = np.array([
        [center_x, bottom + triangle_gap + triangle_size],
        [center_x - triangle_size, bottom + triangle_gap],
        [center_x + triangle_size, bottom + triangle_gap]
    ], dtype=np.int32)

    # Left triangle
    left_triangle = np.array([
        [left - triangle_gap - triangle_size, center_y],
        [left - triangle_gap, center_y - triangle_size],
        [left - triangle_gap, center_y + triangle_size]
    ], dtype=np.int32)

    # Right triangle
    right_triangle = np.array([
        [right + triangle_gap + triangle_size, center_y],
        [right + triangle_gap, center_y - triangle_size],
        [right + triangle_gap, center_y + triangle_size]
    ], dtype=np.int32)


    # Determine each triangle's colour
    left_colour = (
        active_colour_wr if wrist_turn == "left"
        else inactive_colour
    )

    right_colour = (
        active_colour_wr if wrist_turn == "right"
        else inactive_colour
    )

    up_colour = (
        active_colour_ud if up_down == "up"
        else inactive_colour
    )

    down_colour = (
        active_colour_ud if up_down == "down"
        else inactive_colour
    )

    cv2.fillPoly(overlay, [up_triangle], up_colour)
    cv2.fillPoly(overlay, [down_triangle], down_colour)
    cv2.fillPoly(overlay, [left_triangle], left_colour)
    cv2.fillPoly(overlay, [right_triangle], right_colour)

    # Make the box and triangles transparent
    alpha = 0.4
    cv2.addWeighted(
        overlay,
        alpha,
        image,
        1 - alpha,
        0,
        image
    )
    
    cv2.putText(image, "MANUAL  MODE ACTIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(image, f"X: {manual_xyz['x']} | Y: {manual_xyz['y']} | Z: {manual_xyz['z']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(image, "Press 'k' to return to hand tracking", (20, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    base_rotation_x(manual_xyz, image)
    
    terminal_state = {
        "left_hand": {
            "grab": grab,
            "wrist": {
                "up_down": up_down,
                "left_right_rotation": wrist_turn
            },
        },
        "right_hand": {
            "xyz": manual_xyz
        },
    }
    print(json.dumps(terminal_state), flush=True)

    return image


def handle_manual_input(key, input_coordinates, up_down, wrist_turn, grab):
    """Processes keyboard input specifically for manual mode."""
    xy_speed = 50
    z_speed = 20

    if key == 63232:          # Up arrow
        input_coordinates["y"] -= xy_speed
    elif key == 63233:        # Down arrow
        input_coordinates["y"] += xy_speed
    elif key == 63234:        # Left arrow
        input_coordinates["x"] -= xy_speed
    elif key == 63235:        # Right arrow
        input_coordinates["x"] += xy_speed

    elif key == ord("h") and input_coordinates["z"] < 200:
        input_coordinates["z"] += z_speed
    elif key == ord("j") and input_coordinates["z"] > 0:
        input_coordinates["z"] -= z_speed

    elif key == ord("w"):
        if up_down == "center":
            up_down = "up"
        elif up_down == "down":
            up_down = "center"
    elif key == ord("s"):
        if up_down == "center":
            up_down = "down"
        elif up_down == "up":
            up_down = "center"

    elif key == ord("a"):
        if wrist_turn == "center":
            wrist_turn = "left"
        elif wrist_turn == "right":
            wrist_turn = "center"
    elif key == ord("d"):
        if wrist_turn == "center":
            wrist_turn = "right"
        elif wrist_turn == "left":
            wrist_turn = "center"

    elif key == ord("g"):
        if grab == True:
            grab = False
        elif grab == False:
            grab = True
            
    return input_coordinates, up_down, wrist_turn, grab