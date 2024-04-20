from insightface.app import FaceAnalysis

import cv2
from pathlib import Path

class FaceAnalysisWrapper:
    def __init__(
        self,
        name,
        root,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        det_size=256
        ):
        self.application = FaceAnalysis(
            name=name,
            root=root,
            providers=providers)

        self.application.prepare(ctx_id=0, det_size=(det_size, det_size))

    def get_face(self, bgr_format_image):
        return self.application.get(bgr_format_image)