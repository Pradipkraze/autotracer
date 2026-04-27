# -*- coding: utf-8 -*-
"""
QThread worker that runs the processing pipeline off the main GUI thread.
"""
import traceback
import numpy as np

from qgis.PyQt.QtCore import QThread, pyqtSignal


class ProcessingWorker(QThread):
    """
    Runs the raster-to-vector pipeline in a background thread.

    Signals
    -------
    progress(int, str)          : step percentage + message
    finished(list, dict)        : polylines, intermediates
    error(str)                  : error message
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list, dict)
    error = pyqtSignal(str)

    def __init__(self, image_array: np.ndarray, params: dict, parent=None):
        super().__init__(parent)
        self.image_array = image_array
        self.params = params

    def run(self):
        try:
            from .processing import run_pipeline

            polylines, intermediates = run_pipeline(
                self.image_array,
                threshold_method=self.params.get("threshold_method", "otsu"),
                manual_threshold=self.params.get("manual_threshold", 128),
                invert=self.params.get("invert", False),
                smooth_method=self.params.get("smooth_method", "median"),
                smooth_kernel=self.params.get("smooth_kernel", 3),
                smooth_lines=self.params.get("smooth_lines", True),
                dp_epsilon=self.params.get("dp_epsilon", 1.5),
                min_contour_length=self.params.get("min_contour_length", 10),
                progress_callback=lambda step, msg: self.progress.emit(step, msg),
            )
            self.finished.emit(polylines, intermediates)
        except Exception:
            self.error.emit(traceback.format_exc())
