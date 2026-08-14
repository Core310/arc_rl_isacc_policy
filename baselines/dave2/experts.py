import sys
import threading
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ScriptedExpert:
    """
    PD controller that generates expert steering + throttle commands from the vehicle's current lateral error and
    heading error.

    This reads the same telemetry vector that the RL environment provides, specifically the lateral_error (idx 8)
    and heading_error (idx 9) fields.

    The controller is intentionally simple; its not trying to be optimal, just good enough to generate demonstrations
    for behavioral cloning. A BC model that matches this controller's performance has learned basic lane following.
    Our RL model should exceed it.
    """

    def __init__(
        self,
        kp_steer: float = 2.0,
        kd_steer: float = 0.5,
        target_speed: float = 1.5,
        kp_throttle: float = 0.5,
    ):
        """
        Args:
            kp_steer: Proportional gain for steering on lateral_error.
            kd_steer: Derivative gain for steering on heading error.
            target_speed: Desired cruising speed in m/s.
            kp_throttle: Proportional gain for speed control.
        """
        self.kp_steer = kp_steer
        self.kd_steer = kd_steer
        self.target_speed = target_speed
        self.kp_throttle = kp_throttle

    def compute_action(self, telemetry: np.ndarray) -> np.ndarray:
        """
        Compute expert action from current telemetry.

        Args:
            telemetry: 12-element float array matching the vec observation protocol defined in config/experiment.py

        Returns:
            [steering, throttle, brake] in [-1, 1] range.
        """
        # Extract relevant signals (matching TELEMETRY_INDICES)
        speed = telemetry[3]        # IDX_SPEED
        lateral_err = telemetry[8]  # IDX_LAT_ERR
        heading_err = telemetry[9]  # IDX_HDG_ERR

        # PD steering: correct lateral offset and heading angle
        steering = -(self.kp_steer * lateral_err + self.kd_steer * heading_err)
        steering = np.clip(steering, -1.0, 1.0)

        # P throttle: maintain target speed
        speed_err = self.target_speed - speed
        throttle = np.clip(self.kp_throttle * speed_err, 0.0, 1.0)

        # Brake if going too fast
        brake = 0.0
        if speed > self.target_speed * 1.5:
            brake = 0.3
            throttle = 0.0

        return np.array([steering, throttle, brake], dtype=np.float32)


class KeyboardExpert:
    """
    Human-driven expert controller via keyboard input.

    Reads keys on a background daemon thread using raw terminal mode (Linux termios).
    The main env loop calls compute_action() each step, which returns the current command state after applying
    smooth ramping and decay.

    Lifecycle:
        expert = KeyboardExpert()
        expert = expert.compute_action(telemetry) # begins listening (prints controls)
        action = expert.compute_action(telemetry) # call each step
        expert.stop()                             # restores terminal, joins thread

    The telemetry argument is accepted for interface compatibility with ScriptedExpert but is not
    used - the human is the controller.
    """

    # Ramp rates (units per second) - tuned for 10 Hz step rate
    STEER_RAMP_RATE = 2.0     # How fast steering increases while held
    STEER_DECAY_RATE = 3.0    # How fast steering returns to center on release
    THROTTLE_RAMP_RATE = 1.5  # How fast throttle increases while held
    THROTTLE_DECAY_RATE = 2.0 # How fast throttle decays on release
    BRAKE_RAMP_RATE = 3.0     # Brake ramps quickly for safety
    BRAKE_DECAY_RATE = 2.0    # Brake decays on release

    def __init__(self, step_dt: float = 0.1):
        """
        Args:
            step_dt: Expected time between compute_action() calls in (seconds).
                Used to scale ramp rates. Default 0.1 = 10 Hz collection.
        """
        self.step_dt = step_dt

        # Current command state (what compute_action returns)
        self._steering = 0.0
        self._throttle = 0.0
        self._brake = 0.0

        # Key-held state (set True while key is pressed)
        self._key_left = False
        self._key_right = False
        self._key_up = False
        self._key_down = False

        # Control flags
        self._running = False
        self._paused = False
        self.quit_requested = False

        # Thread and terminal state
        self._thread = None
        self._old_terminal_settings = None
        self._lock = threading.Lock()

    def start(self):
        """
        Begin keyboard listening. Switches terminal to raw mode and starts the background key-reader thread.

        Call stop() when done to restore the terminal.
        """
        self._running = True
        self.quit_requested = False

        # Print controls banner
        print("\n" + "=" * 55)
        print("  KEYBOARD TELEOP - Human Expert Data Collection")
        print("=" * 55)
        print("  W / ↑    Throttle     A / ←    Steer left")
        print("  S / ↓    Brake        D / →    Steer right")
        print("  SPACE    Emergency Stop  R      Reset steering")
        print("  P        Pause/resumt    Q      Quit & save")
        print("=" * 55)
        print("  Recording... (press P to pause)\n")

        # Start background key reader
        self._thread = threading.Thread(
            target=self._key_reader_loop,
            daemon=True,
            name="keyboard-teleop",
        )
        self._thread.start()

    def stop(self):
        """Stop keyboard listening and restore terminal settings."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Restore terminal (handled inside _key_reader_loop's finally block,
        # but calls explicitly in case thread died unexpectedly)
        self._restore_terminal()

    def compute_action(self, telemetry: np.ndarray) -> np.ndarray:
        """
        Return current smoothed action from keyboard state.

        Called each env step (~10 Hz). Applies ramping to whatever keys are currently held,
        and decay to released keys. The telemetry arg is accepted for interface compatibility but not used.

        Args:
            telemetry: 12-element float array (unused - human is driving).

        Returns:
            [steering, throttle, brake] in [-1, 1] range.
        """
        with self._lock:
            dt = self.step_dt

            # === Steering: ramp toward direction, decay towards center ===
            if self._key_left and not self._key_right:
                # Steer left (negative steering)
                self._steering -= self.STEER_RAMP_RATE * dt
            elif self._key_right and not self._key_left:
                # Steer right (positive steering)
                self._steering += self.STEER_RAMP_RATE * dt
            else:
                # No steering key held - decay toward center
                if abs(self._steering) < self.STEER_DECAY_RATE * dt:
                    self._steering = 0.0
                elif self._steering > 0:
                    self._steering -= self.STEER_DECAY_RATE * dt
                else:
                    self._steering += self.STEER_DECAY_RATE * dt

            # === Throttle: ramp while held, decay on release ===
            if self._key_up:
                self._throttle += self.THROTTLE_RAMP_RATE * dt
            else:
                self._throttle -= self.THROTTLE_DECAY_RATE * dt

            # === Brake: ramp while held, decay on release ===
            if self._key_down:
                self._brake += self.BRAKE_RAMP_RATE * dt
                self._throttle = 0.0 # Can't throttle and brake simultaneously
            else:
                self._brake -= self.BRAKE_DECAY_RATE * dt

            # Clamp all values
            self._steering = float(np.clip(self._steering, -1.0, 1.0))
            self._throttle = float(np.clip(self._throttle, 0.0, 1.0))
            self._brake = float(np.clip(self._brake, 0.0, 1.0))

        return np.array(
            [self._steering, self._throttle, self._brake], dtype=np.float32
        )

    @property
    def is_paused(self) -> bool:
        """Whether frame recording is paused (P key toggle)."""
        return self._paused

    def status_line(self) -> str:
        """One-line HUD string for terminal output during collection."""
        pause_str = " [PAUSED]" if self._paused else ""
        return (
            f"Steer: {self._steering:+.2f}  "
            f"Thr: {self._throttle:.2f}  "
            f"Brk: {self._brake:.2f}{pause_str}"
        )

    # === Background Key Reader ===
    def _key_reader_loop(self):
        """
        Background thread: read keys in raw terminal mode.

        Uses termios to switch stdin to raw mode (no echo, no line buffering)
        so we get each keypress instantly. Arrow keys arrive as 3-byte escape sequences (ESC [ A/B/C/D]).

        The terminal is restored in the finally block even if the thread crashes or is interrupted.
        """
        try:
            import tty
            import termios
            import select
        except ImportError:
            logger.error(
                "termios/tty not available - keyboard teleop requires Linux. "
                "On Windows, use a gamepad or the scripted expert instead."
            )
            self._running = False
            return

        # Save current terminal settings so we can restore them
        fd = sys.stdin.fileno()
        try:
            self._old_terminal_settings = termios.tcgetattr(fd)
        except termios.error:
            logger.error(
                "Cannot access terminal settings. Are you running in a "
                "terminal with stdin attached? Keyboard teleop won't work "
                "in non-interactive environments (e.g., piped input, IDE "
                "run configs without terminal allocation)."
            )
            self._running = False
            return

        try:
            # Switch to raw mode: instant key reads, no echo
            tty.setraw(fd)

            while self._running:
                # select() with 50ms timeout - responsive but not busy-wait
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)

                if not readable:
                    # No key pressed this cycle - release all held keys
                    # (raw mode doesn't give us key-up events, so we treat
                    # "no key this cycle" as "key released")
                    with self._lock:
                        self._key_left = False
                        self._key_right = False
                        self._key_up = False
                        self._key_down = False
                    continue

                ch = sys.stdin.read(1)

                if ch == 'q' or ch == 'Q':
                    logger.info("Quit requested via keyboard")
                    self.quit_requested = True
                    self._running = False
                    break

                with self._lock:
                    self._handle_key(ch)

        finally:
            # ALWAYS restore terminal - even on crash or KeyboardInterrupt
            self._restore_terminal()

    def _handle_key(self, ch: str):
        """
        Process a single keypress. Must be called with self._lock held.

        Handles both WASD and arrow keys. Arrow keys arrive as 3-byte escape sequences:
        ESC (\x1b) then '[' then A/B/C/D
        """
        # === Arrow key escape sequences ===
        if ch == '\x1b':
            # Read the next two bytes of the escape equence
            try:
                import select as _sel
                # Check if more bytes are availale (they should be for arrow keys. but not for bare ESC press)
                r, _, _ = _sel.select([sys.stdin], [], [], 0.01)
                if r:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        r2, _, _ = _sel.select([sys.stdin], [], [], 0.01)
                        if r2:
                            ch3 = sys.stdin.read(1)
                            if ch3 == 'A':    # Up arrow
                                self._key_up = True
                                return
                            elif ch3 == 'B':  # Down arrow
                                self._key_down = True
                                return
                            elif ch3 == 'C':  # Right arrow
                                self._key_right = True
                                return
                            elif ch3 == 'D':  # Left arrow
                                self._key_left = True
                                return
            except Exception:
                pass
            return # Bare ESC or unrecognized sequence

        # === WASD keys ===
        lower = ch.lower()
        if lower == 'w':
            self._key_up = True
        elif lower == 's':
            self._key_down = True
        elif lower == 'a':
            self._key_left = True
        elif lower == 'd':
            self._key_right = True

        # === Special Keys ===
        elif ch == ' ':
            # Emergency brake - instant full brake, zero throttle
            self._brake = 1.0
            self._throttle = 0.0
            self._key_down = True
        elif lower == 'r':
            # Reset steering to center
            self._steering = 0.0
        elif lower == 'p':
            # Toggle pause
            self._paused = not self._paused
            state = "PAUSED" if self._paused else "RECORDING"
            # Print outside lock would be better but this works fine for feedback
            print(f"\r  [{state}]", end="", flush=True)

    def _restore_terminal(self):
        """Restore terminal to original settings (cooked mode)."""
        if self._old_terminal_settings is not None:
            try:
                import termios
                fd = sys.stdin.fileno()
                termios.tcsetattr(
                    fd, termios.TCSADRAIN, self._old_terminal_settings
                )
                self._old_terminal_settings = None
            except Exception:
                pass # Best effort - don't crash on cleanup
