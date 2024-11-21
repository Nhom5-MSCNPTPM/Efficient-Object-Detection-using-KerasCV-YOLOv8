import keras_cv
import cv2
import numpy as np
import argparse
import os
import time

from infer_utils import decode_detection, plot_boxes, class_names

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    default='inference_data/video_1.mp4',
    help='path to the input video file'
)
args = parser.parse_args()

out_dir = 'outputs'
os.makedirs(out_dir, exist_ok=True)

threshold = 0.25

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_l_backbone_coco"
)
model = keras_cv.models.YOLOV8Detector(
    num_classes=4,
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=3,
)
model.load_weights('model_yolov8large.h5')
print(model.summary())
RESIZE = (640, 640)

cap = cv2.VideoCapture(args.input)
# Get the video's frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = args.input.split(os.path.sep)[-1].split('.')[0]
# Define codec and create VideoWriter object.
out = cv2.VideoWriter(f"{out_dir}/{save_name}.mp4", 
                    cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                    RESIZE)

resizing = keras_cv.layers.JitteredResize(
    target_size=RESIZE,
    scale_factor=(0.75, 1.3),
    bounding_box_format="xyxy",
)

while cap.isOpened:
    ret, frame = cap.read()
    if ret:
        # image_bgr = frame
        frame = cv2.resize(frame, RESIZE)
        image_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_batch = np.expand_dims(image_bgr, 0)

        start_time = time.time()
        output = model.predict(image_batch)
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        boxes, classes, scores = decode_detection(output)

        frame = plot_boxes(frame, boxes, classes, scores, threshold, class_names)
        cv2.putText(
            frame,
            text=f"{fps:.1f} FPS",
            org=(15, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        out.write(frame)
        cv2.imshow('Image', frame)
        # Press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break