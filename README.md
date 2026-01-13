# Peek-a-Boo

A fun interactive app where Reachy Mini plays peek-a-boo with you!

## What It Does

Reachy hides with its head down. When it detects a face, it quickly pops up, wiggles its antennas excitedly, and plays a sound. When you cover your face or look away, Reachy hides again.

## How It Works

1. Reachy starts in a hiding position (head down)
2. Face detection runs continuously using:
   - Reachy's built-in vision (physical robot)
   - Your webcam + OpenCV (simulation mode)
3. When a face is detected: pop up + antenna wiggle + sound
4. When face disappears for ~0.25 seconds: hide again

## Installation

### Clone the Repository

```bash
git clone https://github.com/chandrapavan1104/Peek-a-boo
cd reachy_basic_peek_a_boo
```

### Install Dependencies

```bash
pip install -e .
```

This installs the app and its dependencies (`reachy-mini`, `opencv-python`).

## Usage

### Physical Reachy Mini
1. Install from the dashboard
2. Start the app
3. Stand in front of Reachy and play!

### Simulation (MuJoCo)
Run these commands in order:

**Terminal 1 - Start the simulator:**
```bash
mjpython -m reachy_mini.daemon.app.main --sim
```
Wait for "Daemon started successfully" and the MuJoCo window to appear.

**Terminal 2 - Run the app:**
```bash
python -m reachy_basic_peek_a_boo.main
```

A camera window will open showing your webcam with face detection. Press `Ctrl+C` to stop.

## What To Expect

| You Do | Reachy Does |
|--------|-------------|
| Show face | Pops up fast, wiggles antennas, plays sound |
| Keep looking | Stays up with gentle antenna movements |
| Hide face | Slowly hides back down after ~0.25s |

## Technical Details

- Face detection: Haar Cascade (simulation) / Reachy Vision AI (physical)
- Detection rate: 20 Hz
- Pop-up speed: 0.25 seconds
- Hide speed: 1.0 second
- Sound: Built-in wake_up.wav

## Requirements

**For Physical Reachy Mini:**
- Reachy Mini robot (with built-in camera)

**For Simulation:**
- Python 3.10+
- MuJoCo (included with reachy-mini)
- Webcam (for face detection)
- Dependencies installed via `pip install -e .`:
  - `reachy-mini` - Reachy Mini SDK
  - `opencv-python` - Webcam and face detection

## Note

This app has been tested in simulation (MuJoCo) only. Testing on a physical Reachy Mini is pending. The app auto-detects the environment and should work on physical hardware, but some adjustments may be needed.
