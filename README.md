# Vehicle Counting and Tracking System

This project is a vehicle counting and tracking system using YOLO (You Only Look Once) object detection and a simple tracking algorithm. The system detects and tracks cars, buses, and trucks in a video stream, counts the number of vehicles moving in different directions, and displays the counts on the video frame.

## Features

- Real-time vehicle detection using YOLO object detection.
- Simple tracking algorithm to track the detected vehicles.
- Counts the number of vehicles moving in different directions.
- Displays counts on the video frame.
- Adjustable text size, color, and placement for better visualization.

## Requirements

- Python 3.x
- OpenCV
- Pandas
- Ultralytics YOLO (You can replace it with your preferred YOLO implementation)
- CVZone
- Other dependencies specified in the requirements.txt file

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/maruf9911/Vehicale_Counting_Classes..git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the YOLO model file (`yolov8s.pt`) and class labels file (`coco.txt`) and place them in the project directory.

## Usage

1. Run the main script:

    ```bash
    python vehicle_counting_system.py
    ```

2. The system will process the input video, perform vehicle detection and tracking, and display the results in real-time.

3. Press the 'Esc' key to exit the application.

## Configuration

You can adjust various parameters in the `vehicle_counting_system.py` script to customize the system behavior, such as text size, color, counting lines, etc.

## Acknowledgments

- This project uses the Ultralytics YOLO implementation. Visit [Ultralytics GitHub Repository](https://github.com/ultralytics/yolov5) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
