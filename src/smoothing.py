"""
smoothing.py — Motion smoothing for Aether hand-tracking output.

Implements two complementary strategies:

  1. EMA (Exponential Moving Average) for continuous numeric values
     (XYZ coordinates, wrist roll/pitch angles).

       smoothed = alpha * raw + (1 - alpha) * previous_smoothed

     alpha ∈ (0, 1]:
       • High alpha (e.g. 0.8) → fast response, less smoothing
       • Low  alpha (e.g. 0.2) → heavy smoothing, more lag

     Default: ALPHA_XYZ = 0.3, ALPHA_WRIST = 0.4
     (wrist uses a slightly higher alpha because its downstream consumer,
      classify_roll/classify_pitch, already applies a threshold dead-zone,
      so it can tolerate slightly less pre-smoothing)

  2. Debounce for binary/discrete signals (grab open/close).
     A state change is only confirmed after it holds for N consecutive
     frames, suppressing single-frame flickering.

     Default: GRAB_DEBOUNCE_FRAMES = 3

All parameters are exposed as module-level constants so they can be
adjusted from main2.py without touching this file.
"""

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

ALPHA_XYZ: float = 0.3
"""EMA weight for XYZ coordinate smoothing. Lower = smoother but laggier."""

ALPHA_WRIST: float = 0.4
"""EMA weight for wrist roll/pitch angle smoothing."""

GRAB_DEBOUNCE_FRAMES: int = 3
"""Number of consecutive frames a grab state must hold before it is accepted."""


# ---------------------------------------------------------------------------
# EMA smoother — one instance per signal
# ---------------------------------------------------------------------------

class EMASmoother:
    """
    Single-axis Exponential Moving Average smoother.

    Usage::

        smoother = EMASmoother(alpha=0.3)
        smoothed_x = smoother.update(raw_x)   # call once per frame
        smoother.reset()                        # call when hand disappears
    """

    def __init__(self, alpha: float) -> None:
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._value: float | None = None

    def update(self, raw: float) -> float:
        """Feed a new raw measurement; return the smoothed value."""
        if self._value is None:
            self._value = raw          # first sample: no history yet
        else:
            self._value = self._alpha * raw + (1.0 - self._alpha) * self._value
        return self._value

    def reset(self) -> None:
        """Discard history (call when the hand goes out of frame)."""
        self._value = None

    @property
    def ready(self) -> bool:
        """True once at least one sample has been fed."""
        return self._value is not None


# ---------------------------------------------------------------------------
# XYZ bundle smoother
# ---------------------------------------------------------------------------

class XYZSmoother:
    """
    Three independent EMA smoothers for an XYZ coordinate dict.

    Input / output shape::

        {
            "x": float, "y": float, "z": float | int,
            "pixel_x": int, "pixel_y": int,   # passed through unchanged
        }

    "pixel_x" / "pixel_y" are the raw screen-space pixel positions used
    for drawing overlays. They are NOT smoothed — smoothing pixel positions
    would shift the drawn dot away from the actual detected hand centre.

    Usage::

        smoother = XYZSmoother()
        smoothed = smoother.update(right_xyz)   # returns smoothed dict
        smoother.reset()                         # call when right hand disappears
    """

    def __init__(self, alpha: float = ALPHA_XYZ) -> None:
        self._x = EMASmoother(alpha)
        self._y = EMASmoother(alpha)
        self._z = EMASmoother(alpha)

    def update(self, xyz: dict) -> dict:
        """
        Feed a raw XYZ dict; return a new dict with smoothed values.

        "pixel_x" / "pixel_y" are passed through unchanged — they are pixel
        positions used for drawing and should not be smoothed.
        "z" is returned as int to preserve downstream compatibility.
        """
        return {
            "x":       self._x.update(float(xyz["x"])),
            "y":       self._y.update(float(xyz["y"])),
            "z":       max(0, int(self._z.update(float(xyz["z"] or 0)))),  # clamp: z is never negative
            "pixel_x": xyz["pixel_x"],   # pass through: used for draw overlay
            "pixel_y": xyz["pixel_y"],   # pass through: used for draw overlay
        }

    def reset(self) -> None:
        """Discard all history (call when the tracked hand disappears)."""
        self._x.reset()
        self._y.reset()
        self._z.reset()


# ---------------------------------------------------------------------------
# Wrist angle smoother
# ---------------------------------------------------------------------------

class WristSmoother:
    """
    EMA smoother for the raw roll_delta and pitch_delta angles produced by
    WristDetection.compute_wrist_state().

    Smoothing is applied to the *numeric* deltas **before** classification
    into Left/Right/Up/Down labels, so the thresholds in WristDetection act
    on already-stable values.

    Usage::

        smoother = WristSmoother()
        smoothed = smoother.update(raw_wrist_state)
        smoother.reset()   # call when left hand disappears
    """

    def __init__(self, alpha: float = ALPHA_WRIST) -> None:
        self._roll  = EMASmoother(alpha)
        self._pitch = EMASmoother(alpha)

    def update(self, wrist_state: dict) -> dict:
        """
        Feed a raw wrist_state dict from compute_wrist_state();
        return a new dict with smoothed deltas and re-classified directions.
        """
        from WristDetection import classify_roll, classify_pitch  # avoid circular import at module level

        smoothed_roll  = self._roll.update(wrist_state["roll_delta"])
        smoothed_pitch = self._pitch.update(wrist_state["pitch_delta"])

        return {
            "roll_delta":      smoothed_roll,
            "pitch_delta":     smoothed_pitch,
            "roll_direction":  classify_roll(smoothed_roll),
            "pitch_direction": classify_pitch(smoothed_pitch),
        }

    def reset(self) -> None:
        """Discard history (call when the left hand disappears)."""
        self._roll.reset()
        self._pitch.reset()


# ---------------------------------------------------------------------------
# Grab debounce
# ---------------------------------------------------------------------------

class GrabDebouncer:
    """
    Suppresses single-frame grab flickers by requiring a state change to
    hold for GRAB_DEBOUNCE_FRAMES consecutive frames before it is accepted.

    Output is a string: ``"Grabbing"`` or ``"Open"``, matching the format
    already used in main2.py.

    Usage::

        debouncer = GrabDebouncer()
        stable_grab = debouncer.update(raw_grab_bool)  # "Grabbing" or "Open"
        debouncer.reset()                               # call when hand disappears
    """

    def __init__(self, frames: int = GRAB_DEBOUNCE_FRAMES) -> None:
        self._required = frames
        self._pending_state: bool | None = None
        self._pending_count: int = 0
        self._confirmed_state: bool = False   # last confirmed state

    def update(self, raw: bool) -> str:
        """
        Feed the raw boolean output of is_grabbing();
        return the debounced state as ``"Grabbing"`` or ``"Open"``.
        """
        if raw == self._pending_state:
            self._pending_count += 1
            if self._pending_count >= self._required:
                self._confirmed_state = raw
        else:
            # State changed — start counting from 1
            self._pending_state = raw
            self._pending_count = 1

        return "Grabbing" if self._confirmed_state else "Open"

    def reset(self) -> None:
        """Discard all state (call when the hand disappears)."""
        self._pending_state = None
        self._pending_count = 0
        self._confirmed_state = False