import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from consts import POLYGON, LINE_START, LINE_END, LINE_ZONE, CLASSES

model = YOLO("models/yolo11m.pt")
tracker = sv.ByteTrack(minimum_consecutive_frames=3)
smoother = sv.DetectionsSmoother()
tracker.reset()

polygon_zone = sv.PolygonZone(polygon=POLYGON, triggering_anchors=(sv.Position.CENTER,))
box_annotator = sv.ColorAnnotator()
label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_scale=1.5)
trace_annotator = sv.TraceAnnotator(trace_length=60)
line_zone_annotator = sv.LineZoneAnnotator(text_scale=1.5, text_orient_to_line=True)
line_zone_annotator_multiclass = sv.LineZoneAnnotatorMulticlass(
    text_scale=1.5, text_thickness=2, table_margin=20
)


def main(video_path: str, save_video: bool) -> None:
    print(f"Processing video at path: {video_path}")

    if save_video:
        video_writer = cv2.VideoWriter(
            "data/output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30 * 0.5, (3840, 2160)
        )

    # Generate frames from the video with a specified stride (process every 2nd frame)
    frame_generator = sv.get_video_frames_generator(source_path=video_path, stride=2)

    for i, frame in enumerate(frame_generator):
        print(f"Processing frame {i}")

        # Perform object detection on the frame
        result = model(frame, device="mps", verbose=False, imgsz=1280)[0]

        # Convert detection results to a standardized format
        detections = sv.Detections.from_ultralytics(result)

        # Filter detections that fall within a specified polygonal zone
        detections = detections[polygon_zone.trigger(detections)]

        # Keep only detections belonging to specific classes
        detections = detections[np.isin(detections.class_id, CLASSES)]

        # Assign unique tracker IDs to detections
        detections = tracker.update_with_detections(detections)

        # Apply smoothing to detection coordinates
        detections = smoother.update_with_detections(detections)

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

        # Trigger line crossing detection for specific detections
        LINE_ZONE.trigger(detections=detections)

        # annotate the frame
        annotated_frame = frame.copy()

        # Draw polygon overlay to mark the zone of interest on the frame
        annotated_frame = sv.draw_polygon(
            scene=annotated_frame, polygon=POLYGON, color=sv.Color.RED, thickness=2
        )

        # Add bounding boxes to the annotated frame
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )

        # Add labels to each detected object
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )

        # Draw tracking traces for detected objects
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )

        # Annotate line crossing zone counts on the frame
        annotated_frame = line_zone_annotator.annotate(
            frame=annotated_frame, line_counter=LINE_ZONE
        )

        # Add multiclass labels for line zone crossings
        annotated_frame = line_zone_annotator_multiclass.annotate(
            frame=annotated_frame,
            line_zones=[LINE_ZONE],
            line_zone_labels=["Finish Line"],
        )

        cv2.imshow("Processed Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if save_video:
            video_writer.write(annotated_frame)
            if i % 50 == 0 and i != 0:
                break

    if save_video:
        video_writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", help="Path to the video file", type=str)
    parser.add_argument("--save_video", help="Save the annotated video", type=bool)

    args = parser.parse_args()

    main(args.video_path, args.save_video)
