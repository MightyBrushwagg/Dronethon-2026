### Inter-University Drone Simulation Challenge 2025
Hosted by RHUL Robotics & AI Society 

Date: Saturday, 6th December 2025
Location: Royal Holloway, University of London
Platform: Webots Simulator

### 📌 Overview

Welcome to the official repository for the Inter-University Drone Simulation Challenge 2025.
Teams from across London will compete to build autonomous drones inside the Webots simulator, capable of navigating a custom 3D environment, locating hidden target objects, and completing the task in the shortest possible time.

This repository contains the starter code, sample worlds, object models, and all rules and scoring details needed to prepare for the challenge.
Teams of 2–4 members, all skill levels welcome.

### 📁 Repository Structure
  `drone-sim-challenge-2025/
  │
  ├── starter-code/
  │   ├── basic_controller.py
  │   ├── object_recognition_example.py
  │   ├── simple_search_pattern.py
  │
  ├── worlds/
  │   ├── sample_world_easy.wbt
  │   ├── sample_world_advanced.wbt
  │
  ├── objects/
  │   ├── target_cube.wbo
  │   ├── target_sphere.wbo
  │
  ├── examples/
  │   ├── demo.gif
  │   ├── environment_screenshot.png`

### 🚀 Getting Started
## 1. Install Webots

Download Webots at:
https://cyberbotics.com

## 2. Clone the repository
git clone https://github.com/your-org/drone-sim-challenge-2025.git
cd drone-sim-challenge-2025

## 3. Open a sample world

In Webots:
File → Open World → choose from /worlds

## 4. Attach a controller

Click your drone in the Scene Tree

Set the controller to one from /starter-code

## 5. Begin development

Use the starter templates to build:

navigation logic

search patterns

object detection

AI-style decision-making

### 📜 Rules & Requirements
Team Size

2–4 members

Allowed Languages

Python

C++

MATLAB
(As supported by Webots)

Your Drone Must Be Fully Autonomous

No manual control, no teleoperation.

Allowed Detection Methods

Webots Recognition camera

Colour / shape detection (OpenCV)

Pre-trained ML models

Not Allowed


Internet-dependent models or cloud services

Manual intervention once simulation starts

Provided on Event Day

Sample worlds

Target object definitions


Objective

Locate all designated target objects as quickly and accurately as possible.

### 🧮 Scoring System
## 1. Completion Time (Primary Score)

Faster = higher ranking.
Your final time stops when the drone detects the last required object.

## 2. Detection Accuracy

Correct detections → no penalty

False detections → time penalty

## 3. Stability & Safety

Penalties for:

Crashes

Getting stuck

Excessive oscillation or instability

## 4. Bonus Points

Awarded for:

Smooth navigation

Efficient exploration pattern

Final Score Formula (simplified)
Final Score = Time + Penalties – Bonuses


Lowest score wins.


### 🏛️ Organisers

RHUL Robotics & AI Society
For questions, contact:
Mohamed El Khoukhi
Vice-President, RHUL Robotics & AI Society
📧 yassinekj07@gmail.com

### ⭐ Good Luck!

We can’t wait to see what you build.
Get coding, get creative, and get ready to fly 🚁🔥
