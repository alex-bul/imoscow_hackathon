from ultralytics import YOLO
import torch


def calculate_iou(gt: list[int], pr: list[int]) -> float:
    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    if dx < 0:
        return 0.0
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1
    if dy < 0:
        return 0.0
    overlap_area = dx * dy
    area_gt = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    area_pr = (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1)
    # Calculate union area
    union_area = area_gt + area_pr - overlap_area
    return overlap_area / union_area


class YoloDetector:
    def __init__(self, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        # print('Classes:\n' '\n'.join(self.classes))
        self.model.to(self.device)
        print("Using Device: ", self.device)

    def load_model(self, model_name):
        if model_name:
            model = YOLO(model_name)  # load a custom model
        else:
            model = YOLO("yolov8n.pt")
        return model

    def score_frame(self, frame):
        # width = 640
        # height = 640

        # frame = cv2.resize(frame, (width,height))
        results = self.model(frame, conf=0.5, verbose=False)[0]
        probs, cords = results.probs, results.boxes.xyxyn
        if not probs:
            probs = []
        confidences = [torch.max(p) for p in probs]
        labels = [torch.argmax(p) for p in probs]
        return results, labels, cords, confidences

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, labels, cord, conf, frame, height, width, conf_threshold=0.0):
        detections = []
        for i in range(len(conf)):
            row = cord[i]

            x1, y1 = int(row[0] * width), int(row[1] * height)
            x2, y2 = int(row[2] * width), int(row[3] * height)

            # if self.class_to_label(labels[i]) == 'person':
            confidence = float(cord[i])
            # self.class_to_label
            detections.append(
                ([x1, y1, x2, y2], confidence, self.class_to_label(labels[i]), labels[i])
            )

        return frame, detections