import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from core import filters
from core.histogram import local_histogram_equalization
from core.image_io import load_image, save_image
from core.interpolation import resize, rotate, shear
from core.pipeline import ImagePipeline
from core.utils import display_ready


class WorkbenchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clinical Image Analysis and Enhancement Workbench - Phase 1")
        self.geometry("1220x780")
        self.minsize(1000, 650)
        self.pipeline = ImagePipeline()
        self.metadata = {}
        self.zoom_scale = 1.0
        self.photo = None
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=10)
        controls.grid(row=0, column=0, sticky="ns")

        viewer_wrap = ttk.Frame(self, padding=(0, 10, 10, 10))
        viewer_wrap.grid(row=0, column=1, sticky="nsew")
        viewer_wrap.rowconfigure(0, weight=1)
        viewer_wrap.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(viewer_wrap, bg="#202124", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(viewer_wrap, orient="vertical", command=self.canvas.yview)
        xscroll = ttk.Scrollbar(viewer_wrap, orient="horizontal", command=self.canvas.xview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.status = tk.StringVar(value="Load a DICOM, JPEG, or BMP image to begin.")
        ttk.Label(self, textvariable=self.status, anchor="w", padding=6).grid(row=1, column=0, columnspan=2, sticky="ew")

        ttk.Button(controls, text="Load Image", command=self.load_image).pack(fill="x", pady=(0, 4))
        ttk.Button(controls, text="Save Current Image", command=self.save_current).pack(fill="x", pady=4)
        ttk.Button(controls, text="Undo Last Step", command=self.undo).pack(fill="x", pady=4)
        ttk.Button(controls, text="Reset to Original", command=self.reset).pack(fill="x", pady=4)

        self._section(controls, "Zoom")
        self.interp_method = tk.StringVar(value="bilinear")
        ttk.Combobox(controls, textvariable=self.interp_method, values=["nearest", "bilinear"], state="readonly").pack(fill="x")
        ttk.Button(controls, text="Zoom In", command=lambda: self.zoom(1.25)).pack(fill="x", pady=4)
        ttk.Button(controls, text="Zoom Out", command=lambda: self.zoom(0.8)).pack(fill="x", pady=4)

        self._section(controls, "Spatial Filters")
        self.kernel_size = tk.StringVar(value="3")
        self.variance = tk.StringVar(value="1.0")
        self.edge_operator = tk.StringVar(value="sobel")
        self.edge_direction = tk.StringVar(value="magnitude")
        self._labeled_entry(controls, "Kernel size", self.kernel_size)
        self._labeled_entry(controls, "Gaussian variance", self.variance)
        ttk.Button(controls, text="Average Filter", command=self.apply_average).pack(fill="x", pady=3)
        ttk.Button(controls, text="Gaussian Filter", command=self.apply_gaussian).pack(fill="x", pady=3)
        ttk.Combobox(controls, textvariable=self.edge_operator, values=["sobel", "prewitt"], state="readonly").pack(fill="x", pady=(5, 2))
        ttk.Combobox(controls, textvariable=self.edge_direction, values=["horizontal", "vertical", "magnitude"], state="readonly").pack(fill="x", pady=2)
        ttk.Button(controls, text="Edge Detection", command=self.apply_edge).pack(fill="x", pady=3)
        ttk.Button(controls, text="Median Filter", command=self.apply_median).pack(fill="x", pady=3)

        self._section(controls, "Histogram")
        self.block_size = tk.StringVar(value="8")
        self._labeled_entry(controls, "Local block size", self.block_size)
        ttk.Button(controls, text="Local Equalization", command=self.apply_local_equalization).pack(fill="x", pady=3)

        self._section(controls, "Transforms")
        self.angle = tk.StringVar(value="15")
        self.shear_x = tk.StringVar(value="0.2")
        self.shear_y = tk.StringVar(value="0.0")
        self._labeled_entry(controls, "Rotation angle", self.angle)
        ttk.Button(controls, text="Rotate", command=self.apply_rotation).pack(fill="x", pady=3)
        self._labeled_entry(controls, "Shear X", self.shear_x)
        self._labeled_entry(controls, "Shear Y", self.shear_y)
        ttk.Button(controls, text="Shear", command=self.apply_shear).pack(fill="x", pady=3)

        self._section(controls, "Metadata")
        self.metadata_text = tk.Text(controls, width=34, height=10, wrap="word")
        self.metadata_text.pack(fill="both", expand=True)

        self._section(controls, "Pipeline")
        self.pipeline_list = tk.Listbox(controls, height=6)
        self.pipeline_list.pack(fill="both", expand=True)

    def _section(self, parent, title):
        ttk.Label(parent, text=title, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 4))

    def _labeled_entry(self, parent, label, variable):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label).pack(side="left")
        ttk.Entry(row, textvariable=variable, width=8).pack(side="right")

    def require_image(self):
        if self.pipeline.current is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return None
        return self.pipeline.current

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Medical and image files", "*.dcm *.dicom *.jpg *.jpeg *.bmp"),
                ("DICOM", "*.dcm *.dicom"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return
        try:
            image, metadata = load_image(path)
            self.pipeline.load(image)
            self.metadata = metadata
            self.zoom_scale = 1.0
            self.update_metadata()
            self.update_pipeline_list()
            self.render()
            self.status.set(f"Loaded {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def save_current(self):
        image = self.require_image()
        if image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            save_image(path, image)
            self.status.set(f"Saved {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def update_metadata(self):
        self.metadata_text.delete("1.0", "end")
        for key in sorted(self.metadata):
            self.metadata_text.insert("end", f"{key}: {self.metadata[key]}\n")

    def update_pipeline_list(self):
        self.pipeline_list.delete(0, "end")
        for i, name in enumerate(self.pipeline.operation_names, start=1):
            self.pipeline_list.insert("end", f"{i}. {name}")

    def render(self):
        image = self.require_image()
        if image is None:
            return
        shown = display_ready(image)
        if self.zoom_scale != 1.0:
            shown = resize(shown, self.zoom_scale, self.interp_method.get())
        pil = Image.fromarray(shown)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, pil.width, pil.height))
        self.status.set(f"Image: {image.shape[1]} x {image.shape[0]} | View zoom: {self.zoom_scale:.2f}x")

    def apply_operation(self, name, func):
        image = self.require_image()
        if image is None:
            return
        try:
            result = func(image)
            self.pipeline.apply(result, name)
            self.zoom_scale = 1.0
            self.update_pipeline_list()
            self.render()
            self.status.set(f"Applied {name}")
        except Exception as exc:
            messagebox.showerror("Operation failed", str(exc))

    def int_value(self, variable, label):
        value = int(variable.get())
        if value <= 0:
            raise ValueError(f"{label} must be positive")
        return value

    def float_value(self, variable, label):
        value = float(variable.get())
        return value

    def zoom(self, factor):
        if self.require_image() is None:
            return
        self.zoom_scale = max(0.1, min(8.0, self.zoom_scale * factor))
        self.render()

    def apply_average(self):
        self.apply_operation("Average filter", lambda img: filters.average_filter(img, self.int_value(self.kernel_size, "Kernel size")))

    def apply_gaussian(self):
        self.apply_operation(
            "Gaussian filter",
            lambda img: filters.gaussian_filter(
                img,
                self.int_value(self.kernel_size, "Kernel size"),
                self.float_value(self.variance, "Variance"),
            ),
        )

    def apply_edge(self):
        self.apply_operation(
            f"{self.edge_operator.get().title()} {self.edge_direction.get()} edges",
            lambda img: filters.edge_filter(img, self.edge_operator.get(), self.edge_direction.get()),
        )

    def apply_median(self):
        self.apply_operation("Median filter", lambda img: filters.median_filter(img, self.int_value(self.kernel_size, "Kernel size")))

    def apply_local_equalization(self):
        self.apply_operation(
            "Local histogram equalization",
            lambda img: local_histogram_equalization(img, self.int_value(self.block_size, "Block size")),
        )

    def apply_rotation(self):
        self.apply_operation("Rotation", lambda img: rotate(img, self.float_value(self.angle, "Angle")))

    def apply_shear(self):
        self.apply_operation(
            "Shear",
            lambda img: shear(
                img,
                self.float_value(self.shear_x, "Shear X"),
                self.float_value(self.shear_y, "Shear Y"),
            ),
        )

    def undo(self):
        if self.pipeline.undo():
            self.zoom_scale = 1.0
            self.update_pipeline_list()
            self.render()
            self.status.set("Undid last step")
        else:
            self.status.set("No operation to undo")

    def reset(self):
        if self.pipeline.reset():
            self.zoom_scale = 1.0
            self.update_pipeline_list()
            self.render()
            self.status.set("Reset to original image")
        else:
            self.status.set("No image loaded")

