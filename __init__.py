# -*- coding: utf-8 -*-
"""
Raster Map to Line Vectors Plugin
Converts scanned, georeferenced raster maps into line vectors.
"""

def classFactory(iface):
    from .plugin import RasterToVectorPlugin
    return RasterToVectorPlugin(iface)
