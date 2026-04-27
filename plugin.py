# -*- coding: utf-8 -*-
import os
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsMessageLog, Qgis


class RasterToVectorPlugin:
    """QGIS Plugin: AutoTracer"""

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.action = None
        self.dialog = None

    def initGui(self):
        """Create the menu entry and toolbar icon."""
        icon_path = os.path.join(self.plugin_dir, "icon.png")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()

        self.action = QAction(
            icon,
            "AutoTracer",
            self.iface.mainWindow()
        )
        self.action.setToolTip(
            "Convert a scanned georeferenced raster map into line vector features"
        )
        self.action.triggered.connect(self.run)

        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu("&Raster to Vector", self.action)

    def unload(self):
        """Remove the plugin menu item and icon."""
        self.iface.removePluginRasterMenu("&Raster to Vector", self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.action:
            self.action.deleteLater()

    def run(self):
        """Open the main dialog."""
        try:
            from .dialog import RasterToVectorDialog
            if self.dialog is None:
                self.dialog = RasterToVectorDialog(self.iface)
            self.dialog.show()
            self.dialog.raise_()
            self.dialog.activateWindow()
        except ImportError as e:
            missing = []
            try:
                import cv2
            except ImportError:
                missing.append("opencv-python")
            try:
                import skimage
            except ImportError:
                missing.append("scikit-image")
            try:
                import numpy
            except ImportError:
                missing.append("numpy")

            if missing:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Missing Dependencies",
                    f"The following Python packages are required but not installed:\n\n"
                    f"  • " + "\n  • ".join(missing) +
                    f"\n\nInstall them via OSGeo4W Shell (Windows) or pip:\n"
                    f"  pip install {' '.join(missing)}"
                )
            else:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Plugin Error",
                    f"Failed to load plugin dialog:\n{str(e)}"
                )
