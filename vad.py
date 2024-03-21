import numpy as np
import os
import onnxruntime
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))


class Vad:
    current_time=datetime.now()

    timestamp=current_time.strftime("%H:%M:%S.%f")
    print("Vad time starting time-",timestamp)
    def __init__(self, threshold: float = 0.1):
        model_path = os.path.join(current_dir, "assets", "silero_vad.onnx")

        options = onnxruntime.SessionOptions()
        options.log_severity_level = 4

        self.inference_session = onnxruntime.InferenceSession(
            model_path, sess_options=options
        )
        self.SAMPLING_RATE = 16000
        self.threshold = threshold
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def is_speech(self, audio_data: np.ndarray) -> bool:
        
        input_data = {
            "input": audio_data.reshape(1, -1),
            "sr": np.array([self.SAMPLING_RATE], dtype=np.int64),
            "h": self.h,
            "c": self.c,
        }
        out, h, c = self.inference_session.run(None, input_data)
        self.h, self.c = h, c
        return out > self.threshold
