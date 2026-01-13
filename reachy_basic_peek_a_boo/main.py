import logging
import os
import sys
import threading
import time
from enum import Enum

import cv2
import reachy_mini
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose

# Path to built-in sounds
ASSETS_PATH = os.path.join(os.path.dirname(reachy_mini.__file__), "assets")
PEEKABOO_SOUND = os.path.join(ASSETS_PATH, "wake_up.wav")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class PeekABooState(Enum):
    HIDING = "hiding"
    POPPING_UP = "popping_up"
    PEEKING = "peeking"
    HIDING_AGAIN = "hiding_again"


class ReachyBasicPeekABoo(ReachyMiniApp):
    """
    Peek-a-Boo App for Reachy Mini!

    Reachy hides with its head down. When it detects a face,
    it quickly pops up with a "Peek-a-boo!" and wiggles its antennas.
    Once the face disappears, Reachy hides again waiting for the next player.
    """

    # Optional: URL to a custom configuration page for the app
    custom_app_url: str | None = None

    def __init__(self):
        super().__init__()
        self.state = PeekABooState.HIDING
        self.face_detection_confidence = 0.5
        self.face_gone_frames = 0
        self.face_gone_threshold = 5  # ~0.25 seconds of no face before hiding (reduced from 15)

        # OpenCV face detection setup (for simulation mode)
        self.cap = None
        self.face_cascade = None
        self.use_webcam = None  # Auto-detect: None = try Reachy vision first

    def _init_webcam(self):
        """Initialize webcam and face detector for simulation mode."""
        try:
            # Load Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            # Open webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.warning("Could not open webcam, trying camera 1...")
                self.cap = cv2.VideoCapture(1)

            if self.cap.isOpened():
                logger.info("Webcam initialized successfully!")
                # Set lower resolution for faster processing
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Reduce buffer size to minimize lag (get most recent frame)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return True
            else:
                logger.error("Could not open any webcam!")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize webcam: {e}")
            return False

    def _cleanup_webcam(self):
        """Release webcam resources."""
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam released.")

    def _flush_camera_buffer(self):
        """Quick flush - just grab 2-3 frames."""
        if self.cap is None:
            return
        self.cap.grab()
        self.cap.grab()

    def _create_hide_pose(self):
        """Create a dramatic hiding pose - head tucked down."""
        return create_head_pose(pitch=45, yaw=0, roll=0, degrees=True)

    def _create_peek_pose(self):
        """Create the peek-a-boo reveal pose - head up and alert!"""
        return create_head_pose(pitch=-15, yaw=0, roll=0, degrees=True)

    def _wiggle_antennas(self, reachy_mini: ReachyMini, cycles: int = 3):
        """Wiggle antennas excitedly for the peek-a-boo moment."""
        for _ in range(cycles):
            reachy_mini.set_target(antennas=[1.0, -1.0])
            time.sleep(0.15)
            reachy_mini.set_target(antennas=[-1.0, 1.0])
            time.sleep(0.15)
        reachy_mini.set_target(antennas=[0.5, 0.5])  # Happy antenna position

    def _detect_face_webcam(self) -> bool:
        """Detect face using webcam and OpenCV."""
        if self.cap is None or not self.cap.isOpened():
            return False

        try:
            # Grab and discard buffered frames to get the most recent one
            self.cap.grab()
            ret, frame = self.cap.retrieve()
            if not ret:
                return False

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            # Show the camera feed with face rectangles (optional, for debugging)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Peek-a-Boo Camera', frame)
            cv2.waitKey(1)

            return len(faces) > 0

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return False

    def _detect_face_reachy(self, reachy_mini: ReachyMini) -> bool:
        """Detect face using Reachy's built-in vision (for real robot)."""
        try:
            detections = reachy_mini.vision.get_detections()
            return any(
                d.label == "face" and d.confidence > self.face_detection_confidence
                for d in detections
            )
        except Exception:
            return False

    def _detect_face(self, reachy_mini: ReachyMini) -> bool:
        """Check if a face is detected."""
        if self.use_webcam:
            return self._detect_face_webcam()
        else:
            return self._detect_face_reachy(reachy_mini)

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        logger.info("=" * 50)
        logger.info("Peek-a-Boo App Started!")
        logger.info("=" * 50)

        # Auto-detect: Try Reachy's built-in vision first, fall back to webcam
        if self.use_webcam is None:
            try:
                # Test if Reachy's vision system is available
                test_detections = reachy_mini.vision.get_detections()
                self.use_webcam = False
                logger.info("Using Reachy's built-in vision system (physical robot)")
            except Exception:
                # Vision not available, use webcam (simulation mode)
                self.use_webcam = True
                logger.info("Using webcam for face detection (simulation mode)")

        if self.use_webcam:
            if not self._init_webcam():
                logger.error("Failed to initialize webcam. Falling back to Reachy vision.")
                self.use_webcam = False

        logger.info("Reachy is hiding... Show your face to play!")

        # Start in hiding position
        hide_pose = self._create_hide_pose()
        peek_pose = self._create_peek_pose()

        # Move to initial hiding position
        reachy_mini.set_target(antennas=[0, 0])
        reachy_mini.goto_target(head=hide_pose, duration=1.5)
        self.state = PeekABooState.HIDING
        logger.info("Reachy is now hiding! Looking for faces...")

        try:
            while not stop_event.is_set():
                face_detected = self._detect_face(reachy_mini)

                if self.state == PeekABooState.HIDING:
                    # Waiting for a face to appear
                    if face_detected:
                        logger.info("Face detected! PEEK-A-BOO!")
                        self.state = PeekABooState.POPPING_UP

                        # Quick pop up! Fast and surprising
                        reachy_mini.goto_target(head=peek_pose, duration=0.25)

                        # Excited antenna wiggle
                        self._wiggle_antennas(reachy_mini, cycles=3)

                        # Play peek-a-boo sound
                        try:
                            reachy_mini.media.audio.play_sound(PEEKABOO_SOUND)
                            logger.info("Playing peek-a-boo sound!")
                        except Exception as e:
                            logger.warning(f"Audio error: {e}")

                        self.state = PeekABooState.PEEKING
                        self.face_gone_frames = 0

                elif self.state == PeekABooState.PEEKING:
                    # Stay peeking while face is visible
                    if face_detected:
                        self.face_gone_frames = 0
                        # Small happy antenna movements while engaged
                        try:
                            wiggle = 0.3 * (1 if int(time.time() * 4) % 2 == 0 else -1)
                            reachy_mini.set_target(antennas=[0.5 + wiggle, 0.5 - wiggle])
                        except Exception:
                            pass
                    else:
                        self.face_gone_frames += 1

                        # If face has been gone long enough, hide again
                        if self.face_gone_frames >= self.face_gone_threshold:
                            logger.info("Face gone... hiding again!")
                            self.state = PeekABooState.HIDING_AGAIN

                            # Slowly hide back down
                            reachy_mini.set_target(antennas=[0, 0])
                            reachy_mini.goto_target(head=hide_pose, duration=1.0)

                            # Flush camera buffer to get fresh frames
                            self._flush_camera_buffer()

                            logger.info("Ready for another round... show your face!")
                            self.state = PeekABooState.HIDING
                            self.face_gone_frames = 0

                time.sleep(0.05)  # Check at ~20Hz for faster response

        finally:
            # Clean up webcam
            self._cleanup_webcam()

        # Clean exit - return to neutral position
        logger.info("Peek-a-Boo app stopping... Goodbye!")
        reachy_mini.set_target(antennas=[0, 0])
        neutral_pose = create_head_pose(pitch=0, yaw=0, roll=0, degrees=True)
        reachy_mini.goto_target(head=neutral_pose, duration=1.0)


if __name__ == "__main__":
    app = ReachyBasicPeekABoo()
    app.wrapped_run()
