# -*- coding: utf-8 -*-
"""
Export PowerPoint presentation slides to PNG/JPG images using COM automation.
Uses win32com.client.Dispatch("PowerPoint.Application") to export slides
at 1920x1080 Full HD resolution for visual inspection and documentation.
"""

import os
import sys
import argparse
import win32com.client


def export_slides(pptx_path, output_dir, slide_indices=None, formats=("PNG", "JPG"), width=1920, height=1080):
    """
    Export specified or all slides from a PowerPoint presentation.

    Args:
        pptx_path: Path to .pptx file.
        output_dir: Destination directory for exported images.
        slide_indices: Optional list of 1-based slide indices. If None, exports all slides.
        formats: Tuple of formats to export ('PNG', 'JPG').
        width: Image width in pixels (default: 1920).
        height: Image height in pixels (default: 1080).

    Returns:
        List of generated file paths.
    """
    abs_pptx = os.path.abspath(pptx_path)
    abs_out = os.path.abspath(output_dir)
    os.makedirs(abs_out, exist_ok=True)

    if not os.path.isfile(abs_pptx):
        raise FileNotFoundError(f"Presentation not found: {abs_pptx}")

    ppt_app = None
    presentation = None
    generated_files = []

    try:
        ppt_app = win32com.client.Dispatch("PowerPoint.Application")
        presentation = ppt_app.Presentations.Open(abs_pptx, WithWindow=False)
        total_slides = presentation.Slides.Count

        if slide_indices is None:
            indices = list(range(1, total_slides + 1))
        else:
            indices = [i for i in slide_indices if 1 <= i <= total_slides]

        print(f"Exporting {len(indices)} slides from '{abs_pptx}' to '{abs_out}'...")

        for idx in indices:
            slide = presentation.Slides(idx)
            for fmt in formats:
                ext = fmt.lower()
                filename = f"slide_{idx:02d}.{ext}"
                out_path = os.path.join(abs_out, filename)
                slide.Export(out_path, fmt, width, height)
                generated_files.append(out_path)
                file_size = os.path.getsize(out_path)
                print(f"  Slide {idx:02d} -> {filename} ({file_size} bytes)")

        print(f"Successfully exported {len(generated_files)} image files.")
        return generated_files

    finally:
        if presentation is not None:
            try:
                presentation.Close()
            except Exception as e:
                print(f"Warning closing presentation: {e}", file=sys.stderr)
        if ppt_app is not None:
            try:
                ppt_app.Quit()
            except Exception as e:
                print(f"Warning quitting PowerPoint: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Export PowerPoint slides to images via COM automation.")
    parser.add_argument(
        "--pptx",
        default=r"C:\Users\Roza\Desktop\AnekeshD_Progress.pptx",
        help="Path to source PPTX presentation"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ppt_images"),
        help="Target directory for exported slide images"
    )
    parser.add_argument(
        "--slides",
        default="all",
        help="Comma-separated 1-based slide indices to export, or 'all'"
    )
    parser.add_argument(
        "--formats",
        default="PNG,JPG",
        help="Comma-separated image formats (PNG, JPG)"
    )
    parser.add_argument("--width", type=int, default=1920, help="Output width in pixels")
    parser.add_argument("--height", type=int, default=1080, help="Output height in pixels")

    args = parser.parse_args()

    if args.slides.strip().lower() == "all":
        slide_indices = None
    else:
        slide_indices = [int(s.strip()) for s in args.slides.split(",") if s.strip().isdigit()]

    formats = tuple(f.strip().upper() for f in args.formats.split(",") if f.strip())

    export_slides(
        pptx_path=args.pptx,
        output_dir=args.output_dir,
        slide_indices=slide_indices,
        formats=formats,
        width=args.width,
        height=args.height
    )


if __name__ == "__main__":
    main()
