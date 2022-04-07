import itertools as it
import random

import cv2

colors: list = [c for c in it.product([80, 180, 255, 180, 120], repeat=3)]


def detect_object(frame, req_class, confidence, args):
    cfg_path: str = args.cfg
    weights_path: str = args.weights
    names_path: str = args.names

    net = cv2.dnn_DetectionModel(cfg_path, weights_path)
    net.setInputSize(704, 704)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    frame = cv2.resize(frame, dsize=(704, 704), interpolation=cv2.INTER_AREA)

    with open(names_path, "rt") as f:
        names: list = f.read().rstrip("\n").split("\n")

    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    final_color = random.choice(colors)
    text_color = (0, 0, 0)
    if final_color == (0, 0, 0):
        text_color = (255, 255, 255)
    for classId, confidences, box in zip(
        classes.flatten(), confidences.flatten(), boxes
    ):
        label = "%s: %.0f" % (names[classId], confidences * 100) + "%"
        if req_class == "anything" and confidences >= confidence:
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            left, top, width, height = box
            top = max(top, labelSize[1])
            cv2.rectangle(frame, box, colors[classId + 5], thickness=3)
            cv2.rectangle(
                frame,
                (left - 2, top - labelSize[1] - 5),
                (left + labelSize[0] + 4, top - 1),
                colors[classId + 5],
                cv2.FILLED,
            )
            cv2.putText(
                frame, label, (left, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color
            )
        if names[classId] != req_class:
            continue
        elif confidences < confidence:
            continue
        else:
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            left, top, width, height = box

            top = max(top, labelSize[1])
            cv2.rectangle(frame, box, colors[classId + 5], thickness=2)
            cv2.rectangle(
                frame,
                (left - 2, top - labelSize[1] - 5),
                (left + labelSize[0], top - 1),
                colors[classId + 5],
                cv2.FILLED,
            )
            cv2.putText(
                frame, label, (left, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color
            )

    return frame
