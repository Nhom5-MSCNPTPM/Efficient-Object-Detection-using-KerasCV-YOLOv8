from keras_cv import bounding_box

import cv2

class_names = [
    "red",
    "yellow",
    "green",
    "off",
]

color_mapping = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "off": (255, 255, 255)
}

def decode_detection(output):
    output = bounding_box.to_ragged(output)
    boxes = output['boxes'].numpy()[0]
    classes = output['classes'].numpy()[0]
    scores = output['confidence'].numpy()[0]
    return boxes, classes, scores

def plot_boxes(image, boxes, classes, scores, threshold, class_names):
    # The boxes are sorted as per decreasing confidence score.
    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            class_name = class_names[int(classes[i])]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color=color_mapping[class_name], 
                thickness=2,
                lineType=cv2.LINE_AA
            )
            cv2.putText(
                image,
                text=class_name,
                org=((int(box[0]), int(box[1]-5))),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=color_mapping[class_name],
                thickness=2,
                lineType=cv2.LINE_AA
            )
    return image