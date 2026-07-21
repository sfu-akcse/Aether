import cv2


def get_base_rotation_direction(xy_coordinates, threshold=100):
    if xy_coordinates is None:
        return "No hand"
    if xy_coordinates["x"] <= -threshold:
        return "Left"
    if xy_coordinates["x"] >= threshold:
        return "Right"
    return "Center"


def resolve_base_rotation_zone_owners(previous_zone_owners, hand_directions):
    zone_owners = dict(previous_zone_owners)

    for zone_name in ("Left", "Right"):
        active_hands = [
            hand_name
            for hand_name, direction in hand_directions.items()
            if direction == zone_name
        ]
        current_owner = zone_owners.get(zone_name)

        if current_owner in active_hands:
            continue

        zone_owners[zone_name] = active_hands[0] if active_hands else None

    return zone_owners


def get_active_base_rotation_zones(zone_owners):
    return {
        zone_name
        for zone_name, owner in zone_owners.items()
        if owner is not None
    }


def get_combined_base_rotation_output(zone_owners, hand_directions):
    active_zones = get_active_base_rotation_zones(zone_owners)

    if active_zones == {"Left", "Right"}:
        return "Left+Right"
    if active_zones == {"Left"}:
        return "Left"
    if active_zones == {"Right"}:
        return "Right"

    if any(direction == "Center" for direction in hand_directions.values()):
        return "Center"

    return "No hand"


def draw_base_rotation_indicator(direction, image):
    if direction not in {"Left", "Right"}:
        return

    h, w = image.shape[:2]

    box_size = min(h, w)
    x1 = (w - box_size) // 2
    x2 = x1 + box_size

    left_center_x = x1 // 2
    right_center_x = (x2 + w) // 2

    if direction == "Left":
        cv2.arrowedLine(
            image,
            (left_center_x + 40, h // 2),
            (left_center_x - 40, h // 2),
            (0, 255, 0),
            5,
            tipLength=0.4
        )

    elif direction == "Right":
        cv2.arrowedLine(
            image,
            (right_center_x - 40, h // 2),
            (right_center_x + 40, h // 2),
            (0, 255, 0),
            5,
            tipLength=0.4
        )


def base_rotation_x(xy_coordinates, image):
    draw_base_rotation_indicator(get_base_rotation_direction(xy_coordinates), image)


def draw_base_rotation_indicators(active_zones, image):
    for zone_name in sorted(active_zones):
        draw_base_rotation_indicator(zone_name, image)


def border_box(image, alpha=0.3):
    h, w, c = image.shape

    box_size = min(h, w)
    x1 = w // 2 - box_size // 2
    x2 = w // 2 + box_size // 2

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (x1, h), (0, 0, 255), -1)
    cv2.rectangle(overlay, (x2, 0), (w, h), (0, 0, 255), -1)

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    
