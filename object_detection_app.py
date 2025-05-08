import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, Label, Button, Frame, font
import threading
import os

# Define YOLO file paths (relative)
base_path = os.path.dirname(__file__)
weights_path = os.path.join(base_path, "yolov4.weights")
config_path = os.path.join(base_path, "yolov4.cfg")
names_path = os.path.join(base_path, "coco.names")


# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class labels
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the main GUI window
app = tk.Tk()
app.title("Real-Time Object Detection")
app.geometry("640x480")

# UI elements
title_frame = Frame(app)
title_frame.pack(pady=10)
title_label = Label(title_frame, text="Real-Time Object Detection", font=font.Font(size=20, weight='bold'))
title_label.pack()

button_frame = Frame(app)
button_frame.pack(pady=20)

status_frame = Frame(app)
status_frame.pack(side=tk.BOTTOM, fill=tk.X)

status_label = Label(status_frame, text="Status: Waiting for input...", font=font.Font(size=12))
status_label.pack(pady=5)

instructions_label = Label(app, text="Press 'Start Detection' to begin.\nPress 'Q' to stop detection.", font=font.Font(size=12))
instructions_label.pack(pady=10)

creators_frame = Frame(app)
creators_frame.pack(pady=10)
creators_label = Label(creators_frame, text="Object Detection using YOLO", font=font.Font(size=12), justify=tk.LEFT)
creators_label.pack()

def update_status(message):
    status_label.config(text=f"Status: {message}")

# The main detection logic (in a separate thread)
def run_detection():
    update_status("Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        update_status("Webcam not accessible.")
        messagebox.showerror("Error", "Could not access the webcam.")
        return

    update_status("Detection started... Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            update_status("Failed to read from webcam.")
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            update_status("Detection stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()
    update_status("Waiting for input...")

# Threaded call to prevent GUI freezing
def start_detection_thread():
    threading.Thread(target=run_detection).start()

# Start button
start_button = Button(button_frame, text="Start Detection", command=start_detection_thread, bg='green', fg='white', font=font.Font(size=14))
start_button.pack(side=tk.LEFT, padx=10)

# Run GUI loop
app.mainloop()
