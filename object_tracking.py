import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


VIDEO_PATH = "baoding.mov"
OUTPUT_VIDEO = "tracked_baoding_output.mp4"
OUTPUT_TRAJECTORY_IMAGE = "trajectory_plot.png"
OUTPUT_FIRST_FRAME = "first_frame_keypoints.png"


def main():
    video_file = Path(VIDEO_PATH)

    if not video_file.exists():
        print(f"Error: {VIDEO_PATH} was not found in this folder.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video loaded successfully.")
    print(f"FPS: {fps}")
    print(f"Frame size: {width} x {height}")
    print(f"Total frames: {frame_count}")

    ret, first_frame = cap.read()

    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        return

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints
    orb = cv2.ORB_create(nfeatures=25)

    keypoints = orb.detect(prev_gray, None)

    prev_points = np.array(
        [[kp.pt] for kp in keypoints],
        dtype=np.float32
    )

    if len(prev_points) == 0:
        print("No ORB keypoints detected.")
        cap.release()
        return

    print(f"ORB keypoints detected: {len(prev_points)}")

    # Draw first-frame keypoints
    first_frame_keypoints = first_frame.copy()
    for point in prev_points:
        x, y = point.ravel()
        cv2.circle(first_frame_keypoints, (int(x), int(y)), 4, (0, 255, 0), -1)

    cv2.imwrite(OUTPUT_FIRST_FRAME, first_frame_keypoints)

    # Lucas-Kanade Optical Flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # Mask used to draw trajectories
    mask = np.zeros_like(first_frame)

    # Store trajectories for plotting
    trajectories = [[] for _ in range(len(prev_points))]

    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray_frame,
            prev_points,
            None,
            **lk_params
        )

        if next_points is None or status is None:
            print("Tracking stopped: no next points found.")
            break

        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()

            if i < len(trajectories):
                trajectories[i].append((x_new, y_new))

            mask = cv2.line(
                mask,
                (int(x_new), int(y_new)),
                (int(x_old), int(y_old)),
                (0, 255, 0),
                2
            )

            frame = cv2.circle(
                frame,
                (int(x_new), int(y_new)),
                5,
                (0, 0, 255),
                -1
            )

        output_frame = cv2.add(frame, mask)

        cv2.putText(
            output_frame,
            f"Frame: {frame_index}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        out.write(output_frame)

        prev_gray = gray_frame.copy()
        prev_points = good_new.reshape(-1, 1, 2)

        frame_index += 1

    cap.release()
    out.release()

    print(f"Processed video saved as: {OUTPUT_VIDEO}")
    print(f"First-frame keypoint image saved as: {OUTPUT_FIRST_FRAME}")

    # Plot motion trajectories
    plt.figure(figsize=(10, 8))

    for trajectory in trajectories:
        if len(trajectory) > 1:
            xs, ys = zip(*trajectory)
            plt.plot(xs, ys, marker="o", linewidth=1, markersize=2)

    plt.title("Tracked Keypoint Motion Trajectories")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.savefig(OUTPUT_TRAJECTORY_IMAGE, dpi=300)
    plt.close()

    print(f"Trajectory plot saved as: {OUTPUT_TRAJECTORY_IMAGE}")
    print("Object tracking process complete.")


if __name__ == "__main__":
    main()