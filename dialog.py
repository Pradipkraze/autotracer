# -*- coding: utf-8 -*-
"""
Main dialog for the AutoTracer plugin.
"""

import os
import numpy as np

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QProgressBar, QTextEdit, QSizePolicy, QMessageBox,
    QTabWidget, QWidget, QFormLayout, QSlider, QFrame,
)
from qgis.PyQt.QtCore import Qt, QSize
from qgis.PyQt.QtGui import QFont, QPixmap, QImage, QColor

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry,
    QgsPointXY, QgsMapLayerProxyModel, QgsRasterLayer,
    QgsCoordinateReferenceSystem, QgsMessageLog, Qgis,
    QgsWkbTypes,
)
from qgis.gui import QgsMapLayerComboBox

from .worker import ProcessingWorker


class RasterToVectorDialog(QDialog):
    """Plugin main dialog."""

    STYLE = """
        QDialog {
            background-color: #1e2330;
            color: #e0e6f0;
            font-family: 'Segoe UI', sans-serif;
        }
        QGroupBox {
            border: 1px solid #3a4460;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 6px;
            color: #7eb8f7;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QLabel { color: #c8d4e8; }
        QComboBox, QSpinBox {
            background: #2a3248;
            color: #e0e6f0;
            border: 1px solid #3a4460;
            border-radius: 4px;
            padding: 3px 6px;
            min-height: 22px;
        }
        QComboBox:hover, QSpinBox:hover { border-color: #5b7fd4; }
        QCheckBox { color: #c8d4e8; spacing: 6px; }
        QCheckBox::indicator {
            width: 14px; height: 14px;
            border: 1px solid #5b7fd4;
            border-radius: 3px;
            background: #2a3248;
        }
        QCheckBox::indicator:checked { background: #4a7fd4; }
        QPushButton {
            background: #3a5294;
            color: #e8f0ff;
            border: none;
            border-radius: 5px;
            padding: 6px 18px;
            font-weight: bold;
        }
        QPushButton:hover { background: #4a65b8; }
        QPushButton:pressed { background: #2a3f7a; }
        QPushButton:disabled { background: #2a3248; color: #556080; }
        QProgressBar {
            background: #2a3248;
            border: 1px solid #3a4460;
            border-radius: 4px;
            text-align: center;
            color: #7eb8f7;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #3a5294, stop:1 #5b8fd4);
            border-radius: 3px;
        }
        QTextEdit {
            background: #151922;
            color: #7eb8f7;
            border: 1px solid #2a3450;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
            font-size: 11px;
        }
        QTabWidget::pane { border: 1px solid #3a4460; border-radius: 4px; }
        QTabBar::tab {
            background: #252d42;
            color: #8090b0;
            padding: 6px 14px;
            border: 1px solid #3a4460;
            border-bottom: none;
            border-radius: 4px 4px 0 0;
        }
        QTabBar::tab:selected { background: #1e2330; color: #7eb8f7; }
        QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #3a4460; }
    """

    def __init__(self, iface, parent=None):
        super().__init__(parent or iface.mainWindow())
        self.iface = iface
        self.worker = None
        self._intermediates = {}

        self.setWindowTitle("AutoTracer")
        self.setMinimumSize(480, 620)
        self.setStyleSheet(self.STYLE)

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # Header
        hdr = QLabel("AutoTracer")
        hdr.setFont(QFont("Segoe UI", 14, QFont.Bold))
        hdr.setStyleSheet("color:#7eb8f7; padding-bottom:4px;")
        root.addWidget(hdr)

        sub = QLabel("Converts a scanned georeferenced raster map into vector line features "
                     "through a 5-step processing pipeline.")
        sub.setWordWrap(True)
        sub.setStyleSheet("color:#6a7a9a; font-size:11px;")
        root.addWidget(sub)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); root.addWidget(sep)

        tabs = QTabWidget()
        tabs.addTab(self._build_input_tab(), "⚙ Settings")
        tabs.addTab(self._build_preview_tab(), "🔍 Preview")
        root.addWidget(tabs)

        # Progress
        pg_box = QGroupBox("Progress")
        pg_lay = QVBoxLayout(pg_box)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        pg_lay.addWidget(self.progress_bar)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(90)
        pg_lay.addWidget(self.log_box)
        root.addWidget(pg_box)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Pipeline")
        self.btn_run.setMinimumHeight(34)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

    def _build_input_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        # Input layer
        grp_in = QGroupBox("Input Raster Layer")
        f_in = QFormLayout(grp_in)
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        f_in.addRow("Layer:", self.layer_combo)
        lay.addWidget(grp_in)

        # Thresholding
        grp_thr = QGroupBox("Step 1 & 2 · Binarization & Thresholding")
        f_thr = QFormLayout(grp_thr)

        self.combo_thresh = QComboBox()
        self.combo_thresh.addItems(["Otsu (auto)", "Adaptive (Gaussian)", "Manual"])
        self.combo_thresh.currentIndexChanged.connect(self._on_thresh_method_changed)
        f_thr.addRow("Method:", self.combo_thresh)

        self.spin_manual_thresh = QSpinBox()
        self.spin_manual_thresh.setRange(0, 255)
        self.spin_manual_thresh.setValue(128)
        self.spin_manual_thresh.setEnabled(False)
        f_thr.addRow("Manual value:", self.spin_manual_thresh)

        self.chk_invert = QCheckBox("Invert (lines are lighter than background)")
        f_thr.addRow("", self.chk_invert)
        lay.addWidget(grp_thr)

        # Smoothing
        grp_sm = QGroupBox("Step 3 · Smoothing (Noise Removal)")
        f_sm = QFormLayout(grp_sm)

        self.combo_smooth = QComboBox()
        self.combo_smooth.addItems(["Median", "Gaussian"])
        f_sm.addRow("Filter:", self.combo_smooth)

        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(3, 21)
        self.spin_kernel.setSingleStep(2)
        self.spin_kernel.setValue(3)
        self.spin_kernel.setToolTip("Must be odd. Larger = more aggressive smoothing.")
        f_sm.addRow("Kernel size:", self.spin_kernel)
        lay.addWidget(grp_sm)

        # Vectorization
        grp_vec = QGroupBox("Step 5 · Vectorization Options")
        f_vec = QFormLayout(grp_vec)

        self.chk_smooth_lines = QCheckBox("Smooth output lines (Douglas-Peucker)")
        self.chk_smooth_lines.setChecked(True)
        self.chk_smooth_lines.setToolTip(
            "Reduces stair-step pixel artifacts in the output polylines."
        )
        f_vec.addRow("", self.chk_smooth_lines)

        from qgis.PyQt.QtWidgets import QDoubleSpinBox
        self.spin_dp_epsilon = QDoubleSpinBox()
        self.spin_dp_epsilon.setRange(0.5, 10.0)
        self.spin_dp_epsilon.setSingleStep(0.5)
        self.spin_dp_epsilon.setValue(1.5)
        self.spin_dp_epsilon.setDecimals(1)
        self.spin_dp_epsilon.setToolTip(
            "Douglas-Peucker tolerance in pixels.\n"
            "Higher = smoother lines, fewer vertices.\n"
            "Lower = more faithful to pixel path (more stair-steps).\n"
            "Recommended: 1.5–3.0 for typical maps."
        )
        f_vec.addRow("Line smoothing tolerance (px):", self.spin_dp_epsilon)

        self.spin_min_len = QSpinBox()
        self.spin_min_len.setRange(2, 500)
        self.spin_min_len.setValue(10)
        self.spin_min_len.setToolTip("Discard line features shorter than this many pixels.")
        f_vec.addRow("Min feature length (px):", self.spin_min_len)
        lay.addWidget(grp_vec)

        # Output
        grp_out = QGroupBox("Output")
        f_out = QFormLayout(grp_out)
        self.chk_load_intermediates = QCheckBox("Load intermediate raster layers")
        self.chk_load_intermediates.setChecked(False)
        f_out.addRow("", self.chk_load_intermediates)
        lay.addWidget(grp_out)

        # Post-processing
        grp_post = QGroupBox("Step 6 · Post-Processing")
        f_post = QFormLayout(grp_post)

        self.chk_snap = QCheckBox("Snap geometries to layer")
        self.chk_snap.setChecked(True)
        self.chk_snap.setToolTip(
            "Self-snap: input layer and reference layer are the same.\n"
            "Behavior: Prefer aligning nodes, insert extra vertices where required."
        )
        self.chk_snap.toggled.connect(lambda checked: self.spin_snap_tol.setEnabled(checked))
        f_post.addRow("", self.chk_snap)

        self.spin_snap_tol = QDoubleSpinBox()
        self.spin_snap_tol.setRange(0.0001, 99999.9999)
        self.spin_snap_tol.setSingleStep(0.1)
        self.spin_snap_tol.setValue(1.0)
        self.spin_snap_tol.setDecimals(4)
        self.spin_snap_tol.setToolTip(
            "Snap tolerance in map units (metres, degrees, etc.).\n"
            "Vertices within this distance of each other will be snapped together."
        )
        f_post.addRow("Snap tolerance (map units):", self.spin_snap_tol)

        self.chk_fix = QCheckBox("Fix geometries")
        self.chk_fix.setChecked(True)
        self.chk_fix.setToolTip(
            "Runs 'Fix Geometries' after snapping to repair any\n"
            "self-intersections or invalid geometry errors."
        )
        f_post.addRow("", self.chk_fix)

        lay.addWidget(grp_post)

        lay.addStretch()
        return w

    def _build_preview_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        info = QLabel("After running the pipeline, select a processing stage to preview:")
        info.setWordWrap(True)
        info.setStyleSheet("color:#6a7a9a; font-size:11px;")
        lay.addWidget(info)

        self.combo_preview_stage = QComboBox()
        self.combo_preview_stage.addItems(["binary", "smoothed", "skeleton"])
        self.combo_preview_stage.currentTextChanged.connect(self._update_preview)
        lay.addWidget(self.combo_preview_stage)

        self.preview_label = QLabel("(no preview yet)")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(300)
        self.preview_label.setStyleSheet(
            "border:1px solid #3a4460; border-radius:4px; background:#151922;"
        )
        lay.addWidget(self.preview_label)
        return w

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_thresh_method_changed(self, idx):
        self.spin_manual_thresh.setEnabled(idx == 2)

    def _on_run(self):
        layer = self.layer_combo.currentLayer()
        if not layer or not isinstance(layer, QgsRasterLayer):
            QMessageBox.warning(self, "No Layer", "Please select a raster layer.")
            return

        image_array = self._raster_to_array(layer)
        if image_array is None:
            return

        thresh_map = {0: "otsu", 1: "adaptive", 2: "manual"}
        smooth_map = {0: "median", 1: "gaussian"}

        params = {
            "threshold_method": thresh_map[self.combo_thresh.currentIndex()],
            "manual_threshold": self.spin_manual_thresh.value(),
            "invert": self.chk_invert.isChecked(),
            "smooth_method": smooth_map[self.combo_smooth.currentIndex()],
            "smooth_kernel": self.spin_kernel.value(),
            "smooth_lines": self.chk_smooth_lines.isChecked(),
            "dp_epsilon": self.spin_dp_epsilon.value(),
            "min_contour_length": self.spin_min_len.value(),
            "snap": self.chk_snap.isChecked(),
            "snap_tolerance": self.spin_snap_tol.value(),
            "fix_geometries": self.chk_fix.isChecked(),
        }

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.log_box.clear()
        self.progress_bar.setValue(0)
        self._params = params

        self.worker = ProcessingWorker(image_array, params)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(lambda pl, iv: self._on_finished(pl, iv, layer))
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self._log("⚠ Cancelled by user.")
            self._reset_buttons()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self._log(msg)

    def _on_finished(self, polylines, intermediates, source_layer):
        self._intermediates = intermediates
        self._reset_buttons()

        if not polylines:
            self._log("⚠ No line features were extracted. Try adjusting threshold or smoothing.")
            return

        self._log(f"✔ Creating vector layer with {len(polylines)} features…")
        vl = self._create_vector_layer(polylines, source_layer)

        if self._params.get("snap"):
            vl = self._snap_geometries(vl, self._params["snap_tolerance"])

        if self._params.get("fix_geometries"):
            vl = self._fix_geometries(vl)

        QgsProject.instance().addMapLayer(vl)
        self._log(f"✔ Layer '{vl.name()}' added to project with {vl.featureCount()} features.")

        if self.chk_load_intermediates.isChecked():
            self._load_intermediate_layers(intermediates, source_layer)

        self._update_preview(self.combo_preview_stage.currentText())
        self._log("✔ All done.")

    def _on_error(self, msg):
        self._reset_buttons()
        self._log(f"✖ Error:\n{msg}")
        QMessageBox.critical(self, "Processing Error", msg)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _log(self, text):
        self.log_box.append(text)
        QgsMessageLog.logMessage(text, "RasterToVector", Qgis.Info)

    def _reset_buttons(self):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _raster_to_array(self, layer: QgsRasterLayer) -> np.ndarray:
        """Read a QGIS raster layer into an HxWxC uint8 numpy array."""
        try:
            from qgis.core import QgsRasterBlock, QgsRectangle
            from osgeo import gdal

            path = layer.dataProvider().dataSourceUri()
            ds = gdal.Open(path)
            if ds is None:
                raise RuntimeError(f"GDAL could not open: {path}")

            bands = min(ds.RasterCount, 3)
            arrays = []
            for b in range(1, bands + 1):
                band = ds.GetRasterBand(b)
                arr = band.ReadAsArray().astype(np.float32)
                # normalise to 0-255
                mn, mx = arr.min(), arr.max()
                if mx > mn:
                    arr = (arr - mn) / (mx - mn) * 255.0
                arrays.append(arr.astype(np.uint8))

            if len(arrays) == 1:
                return arrays[0]          # H×W
            return np.stack(arrays, axis=2)  # H×W×C
        except Exception as e:
            QMessageBox.critical(self, "Raster Read Error", str(e))
            return None

    def _create_vector_layer(self, polylines, source_layer):
        """Build a QGIS memory vector layer from pixel-coord polylines. Returns the layer."""
        crs = source_layer.crs()
        extent = source_layer.extent()
        width = source_layer.width()
        height = source_layer.height()

        vl = QgsVectorLayer(f"LineString?crs={crs.authid()}", "Extracted Lines", "memory")
        pr = vl.dataProvider()
        vl.startEditing()

        feats = []
        for pts in polylines:
            # Convert pixel (col, row) → map coordinates
            geo_pts = []
            for col, row in pts:
                x = extent.xMinimum() + (col / width) * extent.width()
                y = extent.yMaximum() - (row / height) * extent.height()
                geo_pts.append(QgsPointXY(x, y))

            if len(geo_pts) < 2:
                continue
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolylineXY(geo_pts))
            feats.append(feat)

        pr.addFeatures(feats)
        vl.commitChanges()
        self._log(f"✔ Built {len(feats)} raw line features.")
        return vl

    def _snap_geometries(self, vl, tolerance: float):
        """
        Self-snap: input = reference = same layer.
        Behavior 1 = Prefer aligning nodes, insert extra vertices where required.
        Returns snapped layer, or original layer on failure.
        """
        try:
            import processing
            self._log(f"⚙ Snapping geometries (tolerance = {tolerance} map units)…")
            result = processing.run(
                "qgis:snapgeometries",
                {
                    "INPUT":            vl,
                    "REFERENCE_LAYER":  vl,
                    "TOLERANCE":        tolerance,
                    "BEHAVIOR":         1,        # Prefer aligning nodes, insert extra vertices
                    "OUTPUT":           "memory:",
                }
            )
            out = result["OUTPUT"]
            out.setName("Extracted Lines (snapped)")
            self._log(f"✔ Snap complete.")
            return out
        except Exception as e:
            self._log(f"⚠ Snap failed: {e} — skipping.")
            return vl

    def _fix_geometries(self, vl):
        """
        Run native:fixgeometries to repair self-intersections and invalid geometries.
        Returns fixed layer, or original layer on failure.
        """
        try:
            import processing
            self._log("⚙ Fixing geometries…")
            result = processing.run(
                "native:fixgeometries",
                {
                    "INPUT":  vl,
                    "METHOD": 1,          # Structure (most thorough)
                    "OUTPUT": "memory:",
                }
            )
            out = result["OUTPUT"]
            out.setName("Extracted Lines (fixed)")
            self._log(f"✔ Fix geometries complete.")
            return out
        except Exception as e:
            self._log(f"⚠ Fix geometries failed: {e} — skipping.")
            return vl

    def _load_intermediate_layers(self, intermediates, source_layer):
        """Load binary/smoothed/skeleton images as temporary raster layers."""
        try:
            from osgeo import gdal, osr
            import tempfile

            ext = source_layer.extent()
            crs_wkt = source_layer.crs().toWkt()

            for name, arr in intermediates.items():
                with tempfile.NamedTemporaryFile(suffix=f"_{name}.tif", delete=False) as f:
                    path = f.name

                h, w = arr.shape[:2]
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.Create(path, w, h, 1, gdal.GDT_Byte)
                res_x = ext.width() / w
                res_y = ext.height() / h
                ds.SetGeoTransform([ext.xMinimum(), res_x, 0, ext.yMaximum(), 0, -res_y])
                srs = osr.SpatialReference()
                srs.ImportFromWkt(crs_wkt)
                ds.SetProjection(srs.ExportToWkt())
                ds.GetRasterBand(1).WriteArray(arr)
                ds.FlushCache()
                ds = None

                rl = QgsRasterLayer(path, f"[rtv] {name}")
                if rl.isValid():
                    QgsProject.instance().addMapLayer(rl)
        except Exception as e:
            self._log(f"⚠ Could not load intermediate layers: {e}")

    def _update_preview(self, stage: str):
        if stage not in self._intermediates:
            return
        arr = self._intermediates[stage]
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(pix)
