import cv2
import math
import threading

from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)


# Shared gesture data
gesture_data = {
    "hand_detected": False,
    "hands": [],
    "gesture": "NO HAND",
    "x": 0.5,
    "y": 0.5
}

lock = threading.Lock()


# Calculate distance between two points
def calculate_distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# Camera function
def camera_loop():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not open!")
        return

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            # Default data
            current_data = {
                "hand_detected": False,
                "hands": [],
                "gesture": "NO HAND",
                "x": 0.5,
                "y": 0.5
            }

            # Check detected hands
            if results.multi_hand_landmarks:

                detected_hands = []

                for i, hand in enumerate(
                    results.multi_hand_landmarks
                ):

                    landmarks = hand.landmark

                    # Draw hand landmarks
                    mp_draw.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # Hand label
                    hand_label = (
                        results.multi_handedness[i]
                        .classification[0].label
                    )

                    # Index finger position
                    index = landmarks[8]

                    x = index.x
                    y = index.y

                    # Palm center position
                    palm_x = landmarks[9].x
                    palm_y = landmarks[9].y

                    # Finger detection
                    tips = [8, 12, 16, 20]
                    pips = [6, 10, 14, 18]

                    fingers = []

                    for tip, pip in zip(tips, pips):

                        if landmarks[tip].y < landmarks[pip].y:
                            fingers.append(1)

                        else:
                            fingers.append(0)

                    # Thumb detection
                    thumb_tip = landmarks[4]
                    thumb_ip = landmarks[3]
                    thumb_mcp = landmarks[2]

                    thumb_extended = (
                        abs(thumb_tip.x - thumb_mcp.x) > 0.06
                    )

                    # Count extended fingers
                    finger_count = sum(fingers)

                    # Pinch detection
                    distance = calculate_distance(
                        landmarks[4],
                        landmarks[8]
                    )

                    # Gesture recognition
                    if (
                        thumb_extended
                        and finger_count == 0
                        and thumb_tip.y < thumb_ip.y
                    ):

                        gesture = "THUMBS UP"

                    elif distance < 0.05:

                        gesture = "PINCH"

                    elif finger_count == 0 and not thumb_extended:

                        gesture = "FIST"

                    elif finger_count == 4:

                        gesture = "OPEN PALM"

                    else:

                        gesture = "UNKNOWN"

                    # Save individual hand data
                    detected_hands.append({

                        "hand": hand_label,
                        "gesture": gesture,
                        "x": x,
                        "y": y,
                        "palm_x": palm_x,
                        "palm_y": palm_y

                    })

                    # Display gesture on camera
                    cv2.putText(
                        frame,
                        f"{hand_label}: {gesture}",
                        (20, 50 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                # Update data for both hands
                current_data = {

                    "hand_detected": True,
                    "hands": detected_hands,
                    "gesture": detected_hands[0]["gesture"],
                    "x": detected_hands[0]["x"],
                    "y": detected_hands[0]["y"]

                }

            # Update shared data
            with lock:

                gesture_data.update(current_data)

            # Show camera window
            cv2.imshow(
                "AI Dual Hand Controller",
                frame
            )

            # Press Q to quit camera
            if cv2.waitKey(1) & 0xFF == ord("q"):

                break

    cap.release()

    cv2.destroyAllWindows()


# Home page
@app.route("/")
def home():

    return send_from_directory(".", "dna.html")


# Gesture API
@app.route("/gesture")
def get_gesture():

    with lock:

        return jsonify(gesture_data)


# Start application
if __name__ == "__main__":

    camera_thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )

    camera_thread.start()

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False,
        use_reloader=False
    )

#http://127.0.0.1:8000/






















