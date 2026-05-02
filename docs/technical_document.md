# Technical Document - Phase 1

## Project

Clinical Image Analysis and Enhancement Workbench.

## Architecture

The application is organized into GUI and processing modules. The GUI in `gui/app.py` manages image loading, controls, display, metadata, and pipeline state. The processing modules under `core/` contain the custom algorithms required by Phase 1.

## Implemented Modules

| Module | Responsibility |
|---|---|
| `core/image_io.py` | JPEG/BMP loading, saving processed output, DICOM loader dispatch |
| `core/dicom_io.py` | Minimal uncompressed DICOM parser and metadata extraction |
| `core/interpolation.py` | Nearest-neighbor resize, bilinear resize, rotation, shearing |
| `core/convolution.py` | Custom edge-padded 2D convolution |
| `core/filters.py` | Average, Gaussian, Sobel/Prewitt, and median filters |
| `core/histogram.py` | Local histogram equalization from scratch |
| `core/pipeline.py` | Sequential enhancement state, undo, reset |
| `gui/app.py` | Tkinter desktop user interface |

## Built-In Function Policy

The project uses Pillow only for reading/writing standard image formats and converting arrays for GUI display. NumPy is used for array storage. The required image processing operations are implemented manually using loops and explicit mathematical logic.

## Contribution Matrix

| Name | ID | Detailed Contribution |
|---|---|---|
| Student 1 |  | GUI layout, image viewer, load/save controls, metadata panel |
| Student 2 |  | DICOM/JPEG/BMP I/O, metadata extraction, error handling |
| Student 3 |  | Custom interpolation, zooming, rotation, shearing |
| Student 4 |  | Convolution engine, average/Gaussian filters, Sobel/Prewitt edges |
| Student 5 |  | Median filter, local histogram equalization, pipeline, testing |

## Defense Checklist

- Explain how images move from file loading into NumPy arrays.
- Explain why the pipeline always applies operations to the current result.
- Explain the custom convolution loops and edge-padding strategy.
- Explain the difference between nearest-neighbor and bilinear interpolation.
- Explain how Sobel/Prewitt horizontal, vertical, and magnitude outputs are computed.
- Explain how local histogram equalization processes blocks independently.
- Be ready to modify kernel size validation, padding behavior, or interpolation mapping during discussion.

