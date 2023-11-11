from ultralytics import YOLO
import torch


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
        results = self.model(frame, verbose=False)[0]
        probs, cords = results.probs, results.boxes.xyxyn
        if not probs:
            probs = []
        confidences = [torch.max(p) for p in probs]
        labels = [torch.argmax(p) for p in probs]
        return results, labels, cords, confidences


    def class_to_label(self, x):
        return self.classes[int(x)]