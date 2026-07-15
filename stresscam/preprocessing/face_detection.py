import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(
        self,
        model_selection=0,
        min_detection_confidence=0.5,
    ):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )


    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        return results
    
    def draw(self, frame, results):
        """
        Draw face bounding box on a copy of the frame.
        """

        frame_out = frame.copy()

        if not results.detections:
            return frame_out

        h, w, _ = frame.shape

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            cv2.rectangle(
                frame_out,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2,
            )

        return frame_out
    def crop(self, frame, results):
        """
        Crop the first detected face.
        """

        if not results.detections:
            return None

        h, w, _ = frame.shape

        bbox = results.detections[0].location_data.relative_bounding_box
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # padding
        pad_top = int(0.18 * bh)
        pad_side = int(0.05 * bw)
        pad_bottom = int(0.02 * bh)

        # expanded box
        x1 = max(0, x - pad_side)
        y1 = max(0, y - pad_top)

        x2 = min(w, x + bw + pad_side)
        y2 = min(h, y + bh + pad_bottom)

        # crop
        face = frame[y1:y2, x1:x2]

        return face