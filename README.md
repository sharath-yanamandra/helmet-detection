# helmet-detection
Here is a real-time safety helmet detection application using a YOLO model and an RTSP stream:

---

# 🪖 Helmet Detection from RTSP Stream using YOLOv8

This project detects whether people are wearing safety helmets in a live video feed (RTSP stream) using the YOLOv8 model from the Ultralytics library. It highlights individuals with/without helmets in different colors and logs statistics in real-time.

## 🚀 Features

- Real-time detection from RTSP video streams
- Differentiates between people **with helmets (green)** and **without helmets (red)**
- Optional saving of the processed video with bounding boxes and stats overlay
- Handles RTSP connection issues with auto-reconnect logic
- Displays current timestamp and detection statistics on screen

## 🧠 Model

The model should be a trained YOLOv8 `.pt` file with at least two classes:
- `With_Helmet`
- `Without_Helmet`

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- NumPy
- [Ultralytics YOLO](https://docs.ultralytics.com)

Install dependencies:

```bash
pip install opencv-python-headless numpy ultralytics
```

## 📁 File Structure

```
.
├── helmet_best1PT.py       # Main script for helmet detection
├── safetyhelmet.ipynb      # (Optional) Notebook for model training or testing
└── README.md               # Documentation
```

## ⚙️ How to Use

Update the following lines in `helmet_best1PT.py`:

```python
model_path = 'path/to/your/best1.pt'
rtsp_url = 'rtsp://username:password@ip_address:port/stream'
```

Then run:

```bash
python helmet_best1PT.py
```

To quit the video stream, press `q`.

## 💾 Optional Output

The processed video with bounding boxes and statistics will be saved automatically with a filename like:

```
helmet_detection_YYYYMMDD_HHMMSS.mp4
```

You can disable this by setting `output_path=None` when initializing the detector.

## 📊 Visualization

- Green boxes: People wearing helmets
- Red boxes: People without helmets
- Overlay shows counts and current timestamp

## 🧯 Error Handling

- Auto-reconnect up to 5 times if RTSP stream fails
- Logs any frame-level or stream-level errors

## 📌 Notes

- Make sure the model file (`best1.pt`) is trained on classes `With_Helmet` and `Without_Helmet`.
- The RTSP stream must be accessible and stable.

## 📝 License

This project is open-source and available under the MIT License.

---
