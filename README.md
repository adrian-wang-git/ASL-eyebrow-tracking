# ASL-eyebrow-tracking

This project implements a system that can detect if a person's eyebrows are raised, neutral, or lowered. 

In American Sign Language (ASL), the eyebrows carry grammar, as eyebrow position is a non-manual marker that can indicate question and question type. Raised brows mark a yes/no question, (as well as topics and conditionals) while lowered/furrowed brows indicate a WH- question (where, what, why, who, how, etc.). 

## Model

This project uses Google's MediaPipe Face Landmarker model without extra training. 

## Installation
* Install [Python 3.12](https://www.python.org/downloads/release/python-31210/) (64-bit Windows installer, check "Add python.exe to PATH")
* Clone this repository
* In the project folder, run `py -3.12 -m venv venv`
* Activate it with `venv\Scripts\activate`
* Install dependencies with `pip install -r requirements.txt`
* Download [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) and put it in a `models` folder
* Run `python brows.py

## Instructions
* Look at the camera rougly centered looking face-on
* Raise or furrow eyebrows, the label and corresponding ASL grammar type will appear at the bottom-left corner
* Press q to quit

## Application components
### brows.py
* Reads frames with OpenCV
* Uses MediaPipe Face Landmarker to get face landmarks and blendshape scores
* Draw eyebrow landmarks
* Combines the blendshapes into weighted average scores, then labels each frame based on hand-tuned thresholds (may need adjusting)

## Notes + Limitations
* Thresholds are constants manually tuned based on each camera angle
* Tilting the head up, down, or side to side (yaw/pitch) can cause inconsistencies in tracking
* The label can flicker during rapid movement but quickly stabilizes