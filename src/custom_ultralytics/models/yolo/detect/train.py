from ultralytics.utils import RANK
from ultralytics.models.yolo.detect import DetectionTrainer

from custom_ultralytics.utils.callbacks.base import replace_with_custom_callbacks


class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add replace wb callback to custom wb callback
        if RANK in {-1, 0}:
            replace_with_custom_callbacks(self)
