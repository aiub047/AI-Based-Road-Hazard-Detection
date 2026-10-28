# AI-Based Road Hazard Detection

A distributed, edge-AI road hazard safety system that detects stopped vehicles and other roadway hazards locally, then warns approaching drivers through highly visible roadside lights, with warnings that propagate upstream as a traffic queue grows.

> **Status:** Proof of concept (started March 2026). Not validated for real-world roadway use.

## Why

A single incident (a crash, disabled vehicle, debris, closed lane, sudden slowdown, ice, or poor visibility) can trigger secondary crashes when approaching drivers don't perceive the danger in time. FHWA multi-state analysis indicates roughly two-thirds of secondary crashes are front-to-rear collisions, and about half occur within 20 minutes and half a kilometer of the original incident. That makes the problem fast and local, and well suited to real-time, infrastructure-based intervention.

## Key Features

- **Local hazard detection**: YOLO-based vision on edge hardware detects stopped vehicles, slowdowns, debris and, in future, large animals such as deer near the roadway.
- **Physical-first warnings**: Roadside warning lights sit in the driver's normal field of view and work for every vehicle, connected or not.
- **Dynamic upstream propagation**: The warning zone follows the estimated tail of the queue instead of staying at the incident site.
- **Distributed nodes**: Modular roadside units share verified hazard events with one another.
- **Multi-channel alerts**: V2I/V2X, mobile and traffic-management integrations act as supporting channels.
- **Confidence-based fusion**: Inputs from multiple nodes, connected-vehicle data, weather feeds and public reports reduce false alerts.
- **Offline-capable**: Time-critical decisions run on the node without a cloud dependency.

## How It Works

```
Camera / Radar / Sensors
        │
        ▼
 Edge AI inference (YOLO) ──► Lane & motion analysis ──► Hazard confidence
        │                                                      │
        ▼                                                      ▼
 Local warning light  ◄──────────  Upstream propagation  ──► Peer nodes
                                                               │
                                              V2I/V2X · Mobile · TMC · Emergency services
```

1. Each roadside node watches its lanes and tracks vehicles frame to frame.
2. A vehicle that stays stationary in a monitored lane beyond a threshold is flagged as a hazard.
3. The nearest warning light activates, plus a configurable number of lights upstream of it.
4. As the queue extends, the nodes further upstream activate in turn, and they clear when traffic resumes.

## Proof of Concept

A small three-lane roadway model with a toy vehicle, a Raspberry Pi, a Pi Camera and a YOLO model:

- Vehicle moving: no warning.
- Vehicle stopped: the nearest light activates, along with two upstream lights that simulate the propagating warning zone.

## Repository Layout

| Path | Description |
|------|-------------|
| `lane_monitor.py`, `v2_lane_monitor.py` | Lane-based stopped-vehicle monitoring and warning logic |
| `calibrate_lanes.py`, `v2_calibrate_lanes.py` | Interactive lane-region calibration |
| `lanes.json` | Saved lane definitions |
| `detect.py`, `car_detect_1.py`, `cam_feed_1.py`, `lane_1.py` | Detection and camera experiments |
| `yolo*.pt` | Pretrained YOLO weights used for experiments |
| `POC/` | Proof-of-concept material |
| `Designs/` | System and hardware designs |
| `Docs/` | Supporting documentation |

## Getting Started

### Prerequisites

- Python 3.9+
- A camera (Pi Camera, USB webcam or video file)
- Raspberry Pi for the prototype; NVIDIA Jetson Orin Nano is the target platform

### Setup

```bash
git clone https://github.com/aiub047/AI-Based-Road-Hazard-Detection.git
cd AI-Based-Road-Hazard-Detection
python3 -m venv venv
source venv/bin/activate
pip install ultralytics opencv-python numpy
```

### Run

```bash
# 1. Define lane regions on your camera view (writes lanes.json)
python v2_calibrate_lanes.py

# 2. Start monitoring
python v2_lane_monitor.py
```

Adjust the hardware and threshold settings at the top of each script to match your setup.

## Roadmap

- [x] Raspberry Pi proof of concept with upstream warning simulation
- [ ] Migrate to NVIDIA Jetson Orin Nano
- [ ] Train and refine the model for more hazard types and conditions (night, rain, snow, low visibility)
- [ ] Large-animal (deer) detection near and approaching the roadway
- [ ] Queue-boundary estimation and improved warning propagation
- [ ] Radar and additional sensor integration
- [ ] Node-to-node communication and multi-node confidence fusion
- [ ] V2I/V2X, traffic-management and emergency-services interfaces (standards-based)
- [ ] Weather and severe-hazard inputs (flash flood, landslide, storms)
- [ ] Field pilot at high-risk roadway segments
- [ ] Optional: drone-based incident verification

## Design Principles

- **Modular**: Sensors, models, communications and warning outputs can be upgraded independently.
- **Interoperable**: Prefer standard interfaces over a proprietary platform.
- **Safety-first**: Warning behavior is subject to transportation-engineering and safety validation before any deployment.
- **Core independence**: Roadside sensing, compute and lights work without drones or the cloud.

## Contributing

Issues and pull requests are welcome. Please describe the scenario you tested and include hardware details with any detection or timing changes.

## References

1. USDOT, Vehicle-to-Infrastructure (V2I) Resources, Connected Vehicle program.
2. Proof-of-concept prototype (see `POC/`).
3. USDOT Smart City Challenge, integrated connected-vehicle and smart-sensor transportation networks.
4. FHWA, national multi-state analysis of secondary crashes.

## License

No license has been specified yet. All rights reserved until one is added.
