"""
Debug tool for raster comparison issues.
Helps diagnose why everything is being highlighted.
"""

import numpy as np
import cv2
import fitz
from pathlib import Path
import matplotlib.pyplot as plt


def debug_raster_comparison(pdf1_path: str, pdf2_path: str, page_index: int = 0, dpi: int = 400):
    """
    Create diagnostic visualization showing why differences are detected.

    Args:
        pdf1_path: Path to first PDF
        pdf2_path: Path to second PDF
        page_index: Page number (0-based)
        dpi: Rendering resolution
    """

    # Render both pages
    def render_page(pdf_path, page_idx, dpi_val):
        doc = fitz.open(pdf_path)
        page = doc[page_idx]
        zoom = dpi_val / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        doc.close()
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img, zoom

    img1, zoom = render_page(pdf1_path, page_index, dpi)
    img2, _ = render_page(pdf2_path, page_index, dpi)

    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    print(f"Image 1 dtype: {img1.dtype}, range: [{img1.min()}, {img1.max()}]")
    print(f"Image 2 dtype: {img2.dtype}, range: [{img2.min()}, {img2.max()}]")

    # Check if images are identical
    if np.array_equal(img1, img2):
        print("Images are IDENTICAL (pixel-perfect match)")
        return

    # Calculate differences
    diff_abs = cv2.absdiff(img1, img2)
    print(f"\nAbsolute difference stats:")
    print(f"  Min: {diff_abs.min()}, Max: {diff_abs.max()}, Mean: {diff_abs.mean():.2f}")
    print(f"  Pixels > 0: {(diff_abs > 0).sum()} ({(diff_abs > 0).sum() / diff_abs.size * 100:.2f}%)")
    print(f"  Pixels > 10: {(diff_abs > 10).sum()} ({(diff_abs > 10).sum() / diff_abs.size * 100:.2f}%)")
    print(f"  Pixels > 18: {(diff_abs > 18).sum()} ({(diff_abs > 18).sum() / diff_abs.size * 100:.2f}%)")
    print(f"  Pixels > 50: {(diff_abs > 50).sum()} ({(diff_abs > 50).sum() / diff_abs.size * 100:.2f}%)")

    # Alignment check
    print("\nTesting alignment...")
    h, w = img1.shape[:2]
    blur1 = cv2.GaussianBlur(img1, (5,5), 0)
    blur2 = cv2.GaussianBlur(img2, (5,5), 0)
    warp = np.eye(2, 3, dtype=np.float32)

    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
        _, warp = cv2.findTransformECC(blur1, blur2, warp, cv2.MOTION_EUCLIDEAN, criteria)
        print(f"  Alignment transform:\n{warp}")

        # Check if significant transformation
        translation = np.sqrt(warp[0,2]**2 + warp[1,2]**2)
        rotation = np.arctan2(warp[1,0], warp[0,0]) * 180 / np.pi
        print(f"  Translation: {translation:.2f} pixels")
        print(f"  Rotation: {rotation:.4f} degrees")

        img2_aligned = cv2.warpAffine(img2, warp, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        diff_aligned = cv2.absdiff(img1, img2_aligned)
        print(f"\nAfter alignment:")
        print(f"  Pixels > 18: {(diff_aligned > 18).sum()} ({(diff_aligned > 18).sum() / diff_aligned.size * 100:.2f}%)")
    except Exception as e:
        print(f"  Alignment failed: {e}")
        img2_aligned = img2
        diff_aligned = diff_abs

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title('PDF 1')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2, cmap='gray')
    axes[0, 1].set_title('PDF 2')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(diff_abs, cmap='hot')
    axes[0, 2].set_title(f'Absolute Diff (max: {diff_abs.max()})')
    axes[0, 2].axis('off')

    # Threshold at different levels
    _, th18 = cv2.threshold(diff_abs, 18, 255, cv2.THRESH_BINARY)
    _, th50 = cv2.threshold(diff_abs, 50, 255, cv2.THRESH_BINARY)
    _, th100 = cv2.threshold(diff_abs, 100, 255, cv2.THRESH_BINARY)

    axes[1, 0].imshow(th18, cmap='gray')
    axes[1, 0].set_title(f'Threshold > 18 ({(th18 > 0).sum() / th18.size * 100:.2f}%)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(th50, cmap='gray')
    axes[1, 1].set_title(f'Threshold > 50 ({(th50 > 0).sum() / th50.size * 100:.2f}%)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(th100, cmap='gray')
    axes[1, 2].set_title(f'Threshold > 100 ({(th100 > 0).sum() / th100.size * 100:.2f}%)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_path = Path('raster_diff_debug.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic visualization saved to: {output_path}")
    plt.close()

    # Recommendations
    print("\nRecommendations:")
    diff_percentage = (diff_abs > 18).sum() / diff_abs.size * 100

    if diff_percentage > 80:
        print("  - PDFs are very different or alignment failed")
        print("  - Try increasing threshold from 18 to 50+")
        print("  - Check if PDFs are actually the same document")
    elif diff_percentage > 20:
        print("  - Significant differences detected")
        print("  - Consider increasing threshold to 30-40")
    elif diff_percentage > 5:
        print("  - Moderate differences (might be compression artifacts)")
        print("  - Current threshold (18) might be too sensitive")
        print("  - Try threshold of 25-35")
    else:
        print("  - Minimal differences detected")
        print("  - Current threshold seems appropriate")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python raster_diff_debug.py <pdf1> <pdf2> [page_index] [dpi]")
        print("\nExample:")
        print("  python raster_diff_debug.py old.pdf new.pdf 0 400")
        sys.exit(1)

    pdf1 = sys.argv[1]
    pdf2 = sys.argv[2]
    page = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    dpi_val = int(sys.argv[4]) if len(sys.argv) > 4 else 400

    debug_raster_comparison(pdf1, pdf2, page, dpi_val)
