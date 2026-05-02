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
        self.geometry("1280x820")
        self.minsize(1050, 680)
        self.pipeline = ImagePipeline()
        self.metadata = {}
        self.zoom_scale = 1.0
        self.photo = None
        self.image_name = tk.StringVar(value="No image loaded")
        self.view_info = tk.StringVar(value="Ready")
        self._configure_style()
        self._build_ui()

    def _configure_style(self):
        self.colors = {
            "app": "#e8edf2",
            "surface": "#f8fafc",
            "panel": "#ffffff",
            "border": "#cfd8e3",
            "text": "#17202a",
            "muted": "#5d6b7a",
            "accent": "#16697a",
            "accent_dark": "#0f4d5a",
            "canvas": "#111418",
            "canvas_grid": "#20252b",
        }
        self.configure(bg=self.colors["app"])
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        default_font = ("Segoe UI", 10)
        heading_font = ("Segoe UI", 11, "bold")
        style.configure(".", font=default_font, foreground=self.colors["text"])
        style.configure("TFrame", background=self.colors["app"])
        style.configure("Sidebar.TFrame", background=self.colors["surface"])
        style.configure("Panel.TFrame", background=self.colors["panel"])
        style.configure("Viewer.TFrame", background=self.colors["canvas"])
        style.configure("Header.TLabel", background=self.colors["surface"], foreground=self.colors["text"], font=("Segoe UI", 12, "bold"))
        style.configure("Muted.TLabel", background=self.colors["surface"], foreground=self.colors["muted"])
        style.configure("Panel.TLabel", background=self.colors["panel"], foreground=self.colors["text"])
        style.configure("PanelMuted.TLabel", background=self.colors["panel"], foreground=self.colors["muted"])
        style.configure("Status.TLabel", background="#dbe4ec", foreground=self.colors["text"], padding=(10, 6))
        style.configure("TLabelframe", background=self.colors["panel"], bordercolor=self.colors["border"], relief="solid")
        style.configure("TLabelframe.Label", background=self.colors["panel"], foreground=self.colors["accent_dark"], font=heading_font)
        style.configure("TButton", padding=(10, 7), relief="flat")
        style.configure("Accent.TButton", background=self.colors["accent"], foreground="#ffffff", bordercolor=self.colors["accent_dark"])
        style.map(
            "Accent.TButton",
            background=[("active", self.colors["accent_dark"]), ("pressed", self.colors["accent_dark"])],
            foreground=[("disabled", "#e6eef2")],
        )
        style.configure("TNotebook", background=self.colors["surface"], borderwidth=0, padding=(0, 8, 0, 0))
        style.configure("TNotebook.Tab", padding=(14, 7), background="#dce5ed", foreground=self.colors["text"])
        style.map("TNotebook.Tab", background=[("selected", self.colors["panel"])])
        style.configure("TEntry", fieldbackground="#ffffff", padding=4)
        style.configure("TCombobox", fieldbackground="#ffffff", padding=4)

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self, style="Sidebar.TFrame", padding=(12, 12, 12, 10), width=350)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        ttk.Label(sidebar, text="Clinical Workbench", style="Header.TLabel").pack(anchor="w")
        ttk.Label(sidebar, text="Phase 1 image enhancement controls", style="Muted.TLabel").pack(anchor="w", pady=(0, 10))

        quick = ttk.Frame(sidebar, style="Sidebar.TFrame")
        quick.pack(fill="x")
        ttk.Button(quick, text="Load Image", style="Accent.TButton", command=self.load_image).pack(fill="x", pady=(0, 6))
        ttk.Button(quick, text="Save Current Image", command=self.save_current).pack(fill="x", pady=3)
        row = ttk.Frame(quick, style="Sidebar.TFrame")
        row.pack(fill="x", pady=3)
        ttk.Button(row, text="Undo", command=self.undo).pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(row, text="Reset", command=self.reset).pack(side="left", fill="x", expand=True, padx=(3, 0))

        tabs = ttk.Notebook(sidebar)
        tabs.pack(fill="both", expand=True, pady=(8, 0))
        image_tab = self._scrollable_tab(tabs, "Image")
        filters_tab = self._scrollable_tab(tabs, "Filters")
        transforms_tab = self._scrollable_tab(tabs, "Transforms")
        info_tab = self._scrollable_tab(tabs, "Info")

        viewer_wrap = ttk.Frame(self, padding=(12, 12, 12, 8))
        viewer_wrap.grid(row=0, column=1, sticky="nsew")
        viewer_wrap.rowconfigure(1, weight=1)
        viewer_wrap.columnconfigure(0, weight=1)

        viewer_header = ttk.Frame(viewer_wrap)
        viewer_header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        viewer_header.columnconfigure(0, weight=1)
        ttk.Label(viewer_header, textvariable=self.image_name, font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(viewer_header, textvariable=self.view_info, foreground=self.colors["muted"]).grid(row=1, column=0, sticky="w")
        ttk.Button(viewer_header, text="Fit", command=self.fit_to_window).grid(row=0, column=1, rowspan=2, sticky="e", padx=(8, 4))
        ttk.Button(viewer_header, text="100%", command=self.actual_size).grid(row=0, column=2, rowspan=2, sticky="e")

        canvas_shell = ttk.Frame(viewer_wrap, style="Viewer.TFrame", padding=1)
        canvas_shell.grid(row=1, column=0, sticky="nsew")
        canvas_shell.rowconfigure(0, weight=1)
        canvas_shell.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_shell, bg=self.colors["canvas"], highlightthickness=0, borderwidth=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(canvas_shell, orient="vertical", command=self.canvas.yview)
        xscroll = ttk.Scrollbar(canvas_shell, orient="horizontal", command=self.canvas.xview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.status = tk.StringVar(value="Load a DICOM, JPEG, or BMP image to begin.")
        ttk.Label(self, textvariable=self.status, anchor="w", style="Status.TLabel").grid(row=1, column=0, columnspan=2, sticky="ew")

        zoom_frame = self._group(image_tab, "Viewer and Zoom")
        self.interp_method = tk.StringVar(value="bilinear")
        ttk.Label(zoom_frame, text="Interpolation", style="Panel.TLabel").pack(anchor="w")
        ttk.Combobox(zoom_frame, textvariable=self.interp_method, values=["nearest", "bilinear"], state="readonly").pack(fill="x", pady=(2, 8))
        zoom_buttons = ttk.Frame(zoom_frame, style="Panel.TFrame")
        zoom_buttons.pack(fill="x")
        ttk.Button(zoom_buttons, text="Zoom In", command=lambda: self.zoom(1.25)).pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(zoom_buttons, text="Zoom Out", command=lambda: self.zoom(0.8)).pack(side="left", fill="x", expand=True, padx=(3, 0))
        ttk.Button(zoom_frame, text="Fit Image to Viewer", command=self.fit_to_window).pack(fill="x", pady=(8, 0))

        spatial_frame = self._group(filters_tab, "Spatial Filters")
        self.kernel_size = tk.StringVar(value="3")
        self.variance = tk.StringVar(value="1.0")
        self.edge_operator = tk.StringVar(value="sobel")
        self.edge_direction = tk.StringVar(value="magnitude")
        self._labeled_entry(spatial_frame, "Kernel size", self.kernel_size)
        self._labeled_entry(spatial_frame, "Gaussian variance", self.variance)
        ttk.Button(spatial_frame, text="Average Filter", command=self.apply_average).pack(fill="x", pady=(8, 3))
        ttk.Button(spatial_frame, text="Gaussian Filter", command=self.apply_gaussian).pack(fill="x", pady=3)
        ttk.Label(spatial_frame, text="Edge operator", style="Panel.TLabel").pack(anchor="w", pady=(10, 0))
        ttk.Combobox(spatial_frame, textvariable=self.edge_operator, values=["sobel", "prewitt"], state="readonly").pack(fill="x", pady=(2, 6))
        ttk.Label(spatial_frame, text="Edge output", style="Panel.TLabel").pack(anchor="w")
        ttk.Combobox(spatial_frame, textvariable=self.edge_direction, values=["horizontal", "vertical", "magnitude"], state="readonly").pack(fill="x", pady=(2, 6))
        ttk.Button(spatial_frame, text="Edge Detection", command=self.apply_edge).pack(fill="x", pady=3)
        ttk.Button(spatial_frame, text="Median Filter", command=self.apply_median).pack(fill="x", pady=3)

        hist_frame = self._group(filters_tab, "Local Histogram")
        self.block_size = tk.StringVar(value="8")
        self._labeled_entry(hist_frame, "Local block size", self.block_size)
        ttk.Button(hist_frame, text="Local Equalization", command=self.apply_local_equalization).pack(fill="x", pady=(8, 0))

        transform_frame = self._group(transforms_tab, "Geometric Transforms")
        self.angle = tk.StringVar(value="15")
        self.shear_x = tk.StringVar(value="0.2")
        self.shear_y = tk.StringVar(value="0.0")
        self._labeled_entry(transform_frame, "Rotation angle", self.angle)
        ttk.Button(transform_frame, text="Rotate", command=self.apply_rotation).pack(fill="x", pady=(8, 10))
        self._labeled_entry(transform_frame, "Shear X", self.shear_x)
        self._labeled_entry(transform_frame, "Shear Y", self.shear_y)
        ttk.Button(transform_frame, text="Shear", command=self.apply_shear).pack(fill="x", pady=(8, 0))

        metadata_frame = self._group(info_tab, "Metadata")
        self.metadata_text = tk.Text(metadata_frame, width=34, height=13, wrap="word", relief="flat", bg="#f5f8fb", fg=self.colors["text"], insertbackground=self.colors["text"])
        self.metadata_text.pack(fill="both", expand=True)
        self.metadata_text.insert("end", "Metadata appears after loading an image.")
        self.metadata_text.configure(state="disabled")

        pipeline_frame = self._group(info_tab, "Pipeline")
        self.pipeline_list = tk.Listbox(pipeline_frame, height=8, relief="flat", bg="#f5f8fb", fg=self.colors["text"], selectbackground=self.colors["accent"], activestyle="none")
        self.pipeline_list.pack(fill="both", expand=True)
        self.pipeline_list.insert("end", "No operations yet")

    def _scrollable_tab(self, notebook, title):
        outer = ttk.Frame(notebook, style="Panel.TFrame")
        canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0, bg=self.colors["panel"])
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, style="Panel.TFrame", padding=(8, 8))
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        notebook.add(outer, text=title)

        def update_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_inner(event):
            canvas.itemconfigure(window_id, width=event.width)

        inner.bind("<Configure>", update_scrollregion)
        canvas.bind("<Configure>", resize_inner)
        canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))
        return inner

    def _group(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=(10, 9))
        frame.pack(fill="x", pady=(0, 10))
        return frame

    def _labeled_entry(self, parent, label, variable):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, style="Panel.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=variable, width=9).pack(side="right")

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
            self.image_name.set(os.path.basename(path))
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
        self.metadata_text.configure(state="normal")
        self.metadata_text.delete("1.0", "end")
        if not self.metadata:
            self.metadata_text.insert("end", "Metadata appears after loading an image.")
        else:
            for key in sorted(self.metadata):
                self.metadata_text.insert("end", f"{key}: {self.metadata[key]}\n")
        self.metadata_text.configure(state="disabled")

    def update_pipeline_list(self):
        self.pipeline_list.delete(0, "end")
        if not self.pipeline.operation_names:
            self.pipeline_list.insert("end", "No operations yet")
            return
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
        info = f"Image: {image.shape[1]} x {image.shape[0]} | View zoom: {self.zoom_scale:.2f}x"
        self.view_info.set(info)
        self.status.set(info)

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

    def actual_size(self):
        if self.require_image() is None:
            return
        self.zoom_scale = 1.0
        self.render()

    def fit_to_window(self):
        image = self.require_image()
        if image is None:
            return
        self.update_idletasks()
        canvas_w = max(1, self.canvas.winfo_width() - 16)
        canvas_h = max(1, self.canvas.winfo_height() - 16)
        img_h, img_w = image.shape[:2]
        self.zoom_scale = max(0.1, min(8.0, min(canvas_w / img_w, canvas_h / img_h)))
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

