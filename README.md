> **Applied AI portfolio project by Brent Oatis, MBA**
> Connecting feature matching and motion tracking to process monitoring and asset visibility.

# CST 544 Week 3 – Object Tracking with Feature Matching and Optical Flow

## Overview

This project demonstrates object tracking using feature matching and optical flow techniques in OpenCV.

A video of Baoding stress balls was used as the input source. The program detects keypoints using the ORB (Oriented FAST and Rotated BRIEF) algorithm and tracks their movement across video frames using the Lucas-Kanade Optical Flow method.

The system visualizes object motion by:

- Detecting keypoints in the first frame
- Tracking feature movement across frames
- Drawing motion trajectories
- Saving a processed tracking video
- Generating trajectory visualizations

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib

---

## Methods Implemented

### ORB Feature Detection

ORB was used to identify distinctive image features that could be tracked throughout the video sequence.

### Lucas-Kanade Optical Flow

The Lucas-Kanade method estimated motion between consecutive frames by tracking feature displacement.

### Motion Visualization

Motion paths were plotted to visualize the trajectories of tracked keypoints over time.

---

## Input File

```
baoding.mov
```

---

## Output Files

```
tracked_baoding_output.mp4
first_frame_keypoints.png
trajectory_plot.png
```

---

## Running the Project

```bash
python object_tracking.py
```

---

## Learning Outcomes

This project demonstrates:

- Keypoint detection
- Feature matching
- Optical flow tracking
- Motion trajectory visualization
- Computer vision object tracking workflows

---

## References

GeeksforGeeks. (2023). Converting color video to grayscale using OpenCV in Python. https://www.geeksforgeeks.org/converting-color-video-to-grayscale-using-opencv-in-python/

GeeksforGeeks. (2023). Feature matching using brute force in OpenCV. https://www.geeksforgeeks.org/feature-matching-using-brute-force-in-opencv/

GeeksforGeeks. (2024). Python OpenCV: Capture video from camera. https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

Siromer. (2024). Detecting and tracking objects with ORB algorithm using OpenCV. Medium. https://medium.com/thedeephub/detecting-and-tracking-objects-with-orb-using-opencv-d228f4c9054e
