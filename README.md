# Clinical Image Analysis and Enhancement Workbench - Phase 1

Desktop medical image processing workbench for Phase 1 of the DIP final project.

## Features

- Load JPEG, BMP, and common uncompressed DICOM images.
- Display image metadata: width, height, bit depth, and DICOM tags.
- Save the current processed image.
- Zoom in/out using custom nearest-neighbor or bilinear interpolation.
- Apply custom spatial operations:
  - Average smoothing
  - Gaussian smoothing with variance input
  - Sobel or Prewitt edge detection
  - Median filtering
  - Local histogram equalization
- Sequential enhancement pipeline:
  - Operations stack on the current result
  - Undo last step
  - Reset to original
- Extra 5-member-team scope:
  - Rotation from scratch using bilinear interpolation
  - Horizontal/vertical shearing from scratch using bilinear interpolation

## Run

```powershell
python main.py
```

If you want to use the bundled Codex Python runtime:

```powershell
& 'C:\Users\AmrNasser\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' main.py
```

## Notes

The image processing algorithms are implemented from scratch in `core/`. Pillow is used only for image file I/O and Tkinter display conversion.

