"""
MultiHandTracker.py — Issue #40: Multi-Hand Support

Tracks left and right hands independently across frames.
Import this into main.py — do not modify main.py's core logic.

Usage:
    from MultiHandTracker import MultiHandTracker, HandSide

    tracker = MultiHandTracker()                               # both hands
    tracker = MultiHandTracker(dominant_hand=HandSide.RIGHT)   # right only
    tracker = MultiHandTracker(is_mirrored=True)               # webcam (flipped)

    # every frame:
    tracker.update(detection_result)
"""

import logging
from enum import Enum

logger = logging.getLogger('AETHER.VISION')

# ── Constants ─────────────────────────────────────────────────────────────────

# At 30fps, 10 frames ≈ 0.3s
HAND_DISAPPEAR_FRAME_THRESHOLD = 10

# Frames required to lock a hand's side on first detection.
# Once locked, side never changes until the hand disappears.
SIDE_LOCK_FRAMES = 8


# ── Data types ────────────────────────────────────────────────────────────────

class HandSide(Enum):
    LEFT  = "Left"
    RIGHT = "Right"


class HandState:
    """Current state of a single tracked hand."""

    def __init__(self, landmarks):
        self.side: HandSide | None          = None
        self.landmarks                      = landmarks
        self.missing_frames                 = 0
        self._lock_history: list[HandSide]  = []
        self._locked                        = False

    def try_lock_side(self, raw: HandSide) -> bool:
        """Accumulate raw guesses. Returns True once side is locked."""
        if self._locked:
            return True
        self._lock_history.append(raw)
        if len(self._lock_history) >= SIDE_LOCK_FRAMES:
            left  = self._lock_history.count(HandSide.LEFT)
            right = self._lock_history.count(HandSide.RIGHT)
            self.side    = HandSide.LEFT if left >= right else HandSide.RIGHT
            self._locked = True
            logger.info("%s hand confirmed and locked", self.side.value)
        return self._locked

    def update(self, landmarks) -> None:
        self.landmarks      = landmarks
        self.missing_frames = 0

    def mark_missing(self) -> None:
        self.missing_frames += 1

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def is_confirmed_gone(self) -> bool:
        return self.missing_frames >= HAND_DISAPPEAR_FRAME_THRESHOLD


# ── Tracker ───────────────────────────────────────────────────────────────────

class MultiHandTracker:
    """Manages detection and tracking of multiple hands across frames.

    Side detection
    --------------
    Uses MediaPipe's handedness classification (not landmark geometry),
    which correctly identifies Left/Right regardless of whether the palm
    or back of the hand faces the camera.

    If is_mirrored=True (webcam with cv2.flip), the handedness label is
    flipped to compensate — MediaPipe sees the pre-flip image so its
    Left/Right is the mirror of what the user sees.

    Side-locking
    ------------
    Once a hand's side is confirmed over SIDE_LOCK_FRAMES frames, it
    never changes while the hand is visible.  The lock releases only when
    the hand disappears for HAND_DISAPPEAR_FRAME_THRESHOLD frames.
    """

    def __init__(self,
                 dominant_hand: HandSide | None = None,
                 is_mirrored: bool = False):
        """
        Parameters
        ----------
        dominant_hand : HandSide or None
            Filter to one hand. None = track both.
        is_mirrored : bool
            True if the frame is flipped with cv2.flip (typical webcam setup).
            Flips the MediaPipe handedness label to match what the user sees.
        """
        self.dominant_hand = dominant_hand
        self.is_mirrored   = is_mirrored
        self._slots: dict[int, HandState]    = {}
        self._hands: dict[HandSide, HandState] = {}

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_dominant_hand(self, side: HandSide | None) -> None:
        self.dominant_hand = side

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_side(self, handedness) -> HandSide:
        """Read side from MediaPipe handedness, flipping if mirrored."""
        label = handedness[0].category_name  # "Left" or "Right"
        side  = HandSide.LEFT if label == "Left" else HandSide.RIGHT
        if self.is_mirrored:
            side = HandSide.RIGHT if side == HandSide.LEFT else HandSide.LEFT
        return side

    def _should_track(self, side: HandSide) -> bool:
        return self.dominant_hand is None or side == self.dominant_hand

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detection_result) -> None:
        """Process one frame of MediaPipe detection results."""
        active_slots: set[int] = set()

        for slot, (hand_landmarks, handedness) in enumerate(
            zip(detection_result.hand_landmarks, detection_result.handedness)
        ):
            raw = self._get_side(handedness)
            active_slots.add(slot)

            if slot not in self._slots:
                self._slots[slot] = HandState(hand_landmarks)

            state = self._slots[slot]
            state.update(hand_landmarks)
            newly_locked = state.try_lock_side(raw)

            if newly_locked and state.side not in self._hands:
                if self._should_track(state.side):
                    self._hands[state.side] = state

        # Age out slots not seen this frame
        for slot in list(self._slots):
            if slot in active_slots:
                continue

            state = self._slots[slot]
            state.mark_missing()

            if state.is_confirmed_gone:
                if state.side and state.side in self._hands:
                    del self._hands[state.side]
                    logger.info("%s hand removed after %d missing frames",
                                state.side.value, HAND_DISAPPEAR_FRAME_THRESHOLD)
                del self._slots[slot]
            else:
                logger.debug("slot %d (%s) occluded — holding (%d/%d)",
                             slot,
                             state.side.value if state.side else "unlocked",
                             state.missing_frames,
                             HAND_DISAPPEAR_FRAME_THRESHOLD)

    def get_active_hands(self) -> list[HandState]:
        return list(self._hands.values())

    def get_hand(self, side: HandSide) -> HandState | None:
        return self._hands.get(side)

    def is_hand_present(self, side: HandSide) -> bool:
        return side in self._hands

    def get_dominant_hand_state(self) -> HandState | None:
        if self.dominant_hand is None:
            return None
        return self._hands.get(self.dominant_hand)