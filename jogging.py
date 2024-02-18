import cv2
import numpy as np
import mediapipe as mp

video_path = "./WHATSAAP ASSIGNMENT.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

BOUNCE_DISTANCE = 30
REACH_DISTANCE = 70

def detect_basketball(frame):
     """
    Detects a basketball in the given frame using color-based segmentation.

    Parameters:
    - frame (numpy.ndarray): Input frame (image) in BGR format.

    Returns:
    - tuple: A tuple containing:
        - basketball_detected (bool): True if a basketball is detected, False otherwise.
        - basketball_position (tuple or None): A tuple (x, y, w, h) representing the bounding box
          of the detected basketball. None if no basketball is detected.
    """

    # Define the HSV color range for the basketball (yellow in this case)
    lower_bound = np.array([20, 100, 100])
    upper_bound = np.array([30, 255, 255])

    # Convert the BGR frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask based on color range
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    basketball_detected = False
    basketball_position = None

    # If contours are found, choose the largest one as the basketball
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        basketball_position = (x, y, w, h)
        basketball_detected = True

    return basketball_detected, basketball_position


def main(cap, fps):
    """
    Main function for processing a video stream, detecting basketball dribbles, and tracking statistics.

    Parameters:
    - cap (cv2.VideoCapture): Video capture object.
    - fps (float): Frames per second of the video stream.

    Returns:
    - None (Results displayed in real-time using OpenCV).
    """
    dribble_count = 0

    start_bound = False

    reach_ground = False

    initial_x = 0
    initial_y = 0

    bound_x = 0
    bound_y = 0

    reach_ground_frame = 0
    start_bound_frame = 0

    fastest_speed = 1e4
    fastest_bound = "None"

    slowest_speed = -1e4
    slowest_bound = "None"

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        basketball_detected, basketball_position = detect_basketball(frame)

        if basketball_detected:
            # Take the current coordinates of the ball
            x_cur, y_cur, w_cur, h_cur = basketball_position

            # If we haven't track initial coordinates of the ball for a bounce
            if not start_bound and not reach_ground:
                if dribble_count == 0:
                    start_bound = True
                    initial_x, initial_y, _, _ = basketball_position
                    start_bound_frame = i
                # Avoid instantenously assignment of the initial ball after bounce is count
                else:
                    if bound_y - y_cur > BOUNCE_DISTANCE + 10:
                        start_bound = True
                        initial_x, initial_y, _, _ = basketball_position
                        start_bound_frame = i

            # If we haven't reach the ground
            if start_bound and not reach_ground:
                # Criteria for reaching the ground
                if y_cur - initial_y > REACH_DISTANCE:
                    reach_ground = True
                    reach_ground_frame = i
                    bound_x = x_cur
                    bound_y = y_cur

            # If we have reach the ground
            if start_bound and reach_ground:
                # Criteria for bounce
                if bound_y - y_cur > BOUNCE_DISTANCE:
                    dribble_count += 1
                    reach_ground = False
                    start_bound = False
                    
                    track_hand_init = False
                    track_hand_bound = False

                    # Fastest and Slowest bounce
                    if abs(i - start_bound_frame) < fastest_speed:
                        fastest_bound = dribble_count
                        fastest_speed = abs(i - start_bound_frame)
                    elif abs(i - start_bound_frame) > slowest_speed:
                        slowest_bound = dribble_count
                        slowest_speed = abs(i - start_bound_frame)

                    reach_ground_frame = i
                    start_bound_frame = i

                    i = 0

            if start_bound:
                cv2.circle(frame, (initial_x, initial_y), 5, (0, 0, 255), -1)

            if reach_ground:
                cv2.circle(frame, (bound_x, bound_y), 5, (255, 0, 0), -1)

            # If we didn't reach the ground or bounce back for too long, re-initialise the initial ball coordiantes
            if ((i - start_bound_frame) >= int(fps+5) and start_bound) or ((i - reach_ground_frame) >= int(fps/2) and reach_ground) or (not reach_ground and not start_bound):
                start_bound = False
                reach_ground = False

            cv2.putText(
                frame, f'Dribble Count: {dribble_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame, f'Fastest : Dribble Number {fastest_bound}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA
            )
            cv2.putText(
                frame, f'Slowest : Dribble Number {slowest_bound}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6a, (255, 255, 255), 1,
                cv2.LINE_AA
            )


            cv2.rectangle(frame, (x_cur, y_cur), (x_cur + w_cur, y_cur + h_cur), (0, 255, 0), 2)

        cv2.imshow('Basketball Dribble Analysis', frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

        i += 1

if __name__ == "__main__":
    main(cap, fps)
    cap.release()
    cv2.destroyAllWindows()
