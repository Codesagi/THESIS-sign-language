#!/usr/bin/env python3
# ============================================================
# tools/generate_marker.py — Generate ArUco Marker
# ============================================================

import cv2
import numpy as np
from pathlib import Path

def generate_aruco_marker(marker_id=0, size_pixels=400):
    """
    Generate ArUco marker image.
    
    Parameters
    ----------
    marker_id : int
        Marker ID (0-249 for DICT_6X6_250)
    size_pixels : int
        Image size in pixels
    
    Returns
    -------
    marker_image : np.ndarray
        Black and white marker image
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels)
    return marker_image


def main():
    print("=" * 60)
    print("ArUco Marker Generator")
    print("=" * 60)
    
    # Generate marker
    marker_id = 0
    size = 400  # pixels
    
    marker = generate_aruco_marker(marker_id, size)
    
    # Save
    output = Path("aruco_marker_id0.png")
    cv2.imwrite(str(output), marker)
    
    print(f"\n✓ Marker generated: {output}")
    print(f"  ID: {marker_id}")
    print(f"  Size: {size}x{size} pixels")
    print(f"\nNext steps:")
    print(f"  1. Print this image at 10cm x 10cm")
    print(f"  2. Place on flat surface (desk)")
    print(f"  3. Run: python main.py → Word-to-Sign → Marker AR")
    print(f"  4. Point camera at marker")
    print("=" * 60)


if __name__ == "__main__":
    main()