import cv2
import mediapipe as mp
import numpy as np


class SkinSegmenter:

    def __init__(
        self,
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    ):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
        )

        self.LEFT_EYE = [
            33, 133, 160, 159, 158, 157, 173, 153, 154, 155, 144, 145
        ]

        self.RIGHT_EYE = [
            362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382
        ]

        self.LIPS = [
            61, 146, 91, 181, 84, 17, 314, 405,
            321, 375, 291, 308, 324, 318, 402,
            317, 14, 87, 178, 88
        ]

    def detect_landmarks(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        return results

    def draw_landmarks(self, frame, results):

        frame_out = frame.copy()

        if not results.multi_face_landmarks:
            return frame_out

        mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles

        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame_out,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
            )

        return frame_out

    def create_face_mask(self, frame, results):

        if not results.multi_face_landmarks:
            return None

        h, w = frame.shape[:2]

        landmarks = results.multi_face_landmarks[0]

        points = []

        for lm in landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        hull = cv2.convexHull(points)

        mask = np.zeros((h, w), dtype=np.uint8)

        cv2.fillConvexPoly(mask, hull, 255)

        return mask

    def apply_mask(self, frame, mask):

        return cv2.bitwise_and(frame, frame, mask=mask)

    def remove_features(self, mask, results):

        h, w = mask.shape

        landmarks = results.multi_face_landmarks[0]

        for region in [self.LEFT_EYE, self.RIGHT_EYE, self.LIPS]:

            pts = []

            for idx in region:
                lm = landmarks.landmark[idx]
                pts.append([int(lm.x*w), int(lm.y*h)])

            pts = np.array(pts, np.int32)

            cv2.fillConvexPoly(mask, pts, 0)

        return mask
    
    def segment(self, frame):
        """
        Complete skin segmentation pipeline.

        Returns
        -------
        mask : np.ndarray
            Binary skin mask.
        """

        results = self.detect_landmarks(frame)

        if not results.multi_face_landmarks:
            return None, None

        mask = self.create_face_mask(frame, results)
        mask = self.remove_features(mask, results)

        return mask, results