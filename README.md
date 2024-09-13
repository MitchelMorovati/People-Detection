# People-Detection
# Airport People Tracking with YOLOv8 and OpenCV

This project is a recreation of the people tracking system from scratch, based on a concept presented in [this YouTube video](https://www.youtube.com/watch?v=COYUiWxthMc). Since the original source did not provide code, I implemented the system entirely from scratch, incorporating additional functionalities to meet the project requirements.

## Overview

This project uses YOLOv8 for region of interest (ROI) detection and OpenCV for tracking and counting people based on flow direction in an airport setting. The key features include:

## Key Features

- **Region of Interest (ROI) Detection:**
  - Detects people within predefined zones of interest in the airport.
  - Ensures high accuracy in detecting individuals in specific zones.

- **Flow Arrow Counting:**
  - Tracks the movement direction of individuals using OpenCV.
  - Counts the number of people passing through ROIs based on movement direction.

## Technologies Used

- **YOLOv8:** A state-of-the-art object detection model used for identifying and tracking individuals in the scene.
- **OpenCV:** Employed to track movement using flow arrows and count individuals passing through the ROIs.
- **Python:** The primary language for implementing YOLOv8 and OpenCV functionality.

[Watch the video made from this script!](output.mp4)
