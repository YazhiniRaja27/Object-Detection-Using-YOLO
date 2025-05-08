# YOLOv4 Real-Time Object Detection
This project implements real-time object detection using the YOLOv4 model and OpenCV. It includes a basic Python script for object detection via webcam and a GUI-based application built with Tkinter.

## 🧠 Overview
- **Model:** YOLOv4
- **Frameworks:** OpenCV, NumPy, Tkinter
- **Input Source:** Webcam
- **Dataset:** COCO (Common Objects in Context)

## 🚀 Features
- Live object detection with bounding boxes and class labels
- GUI interface for starting/stopping detection
- Easy to run and extend

## 📁 Project Structure
├── object_detection.py # Command-line based YOLO detector
├── object_detection_app.py # GUI-based detector using Tkinter
├── yolov4.weights # Pre-trained weights (Download separately)
├── yolov4.cfg # YOLOv4 configuration file
├── coco.names # Class labels


## 💻 Requirements
- Python 3.7+
- OpenCV
- NumPy
- Tkinter (comes with most Python installations)

Install dependencies:
pip install opencv-python numpy


🧪 How to Run
1. Clone the repository:
git clone https://github.com/your-username/YOLO-Object-Detection.git
cd YOLO-Object-Detection

2. Download YOLOv4 files:
Place the following files in the project directory:
yolov4.weights: https://pjreddie.com/media/files/yolov4.weights
yolov4.cfg: https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
coco.names: https://github.com/pjreddie/darknet/blob/master/data/coco.names

3. Run from terminal:
python object_detection.py

4. Run GUI version:
python object_detection_app.py

Press Q to stop detection.



This project was developed as part of my Semester 3 coursework for the Core Computer Science subject. It gave me hands-on experience with real-time object detection using deep learning models like YOLOv4, as well as working with OpenCV and building a GUI using Tkinter. Through this project, I gained a deeper understanding of how machine learning models interact with real-world input like live video streams, and how to deploy them in user-friendly interfaces.
— Yazhini Raja


# Object-Detection-Using-YOLO
Real-time object detection using YOLOv4 and OpenCV in Python. Includes both command-line and GUI (Tkinter) apps to detect objects via webcam using pretrained COCO weights. Great for learning computer vision and deep learning basics.
