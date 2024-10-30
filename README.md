# AI_traffic_analysis

This project aims to analyze the traffic data on a road detecting cars, trucks, buses and motorcycles. The project consists of the detection of the vehicles and analysis of the traffic data. The project is implemented using **Python**, leveraging packages such as **Supervision**, **OpenCV** and **Ultralytics**.

## Installation

The projects uses Poetry to manage dependencies. All the dependencies are in pyproject.toml. To install the them, run the following command:

```bash
poetry install
```

## Usage

To run the project, run the following command:

```bash
python3 main.py --video_path data/video.mp4
```

## Output video

The example output video looks like:

![Example video](src/data/output.gif)

## Models used

- **YOLO v11** - brand new version of pre-trained YOLO with improved performance released by Ultralytics.