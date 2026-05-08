import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
import threading
from datetime import datetime

# Core imports strictly matching your project structure
from core import frequency, morphology, noise, roi
from core import filters
from core.histogram import local_histogram_equalization, global_histogram_equalization
from core.image_io import load_image, save_image
from core.interpolation import resize, rotate, shear
from core.pipeline import ImagePipeline
from core.utils import display_ready

# Modern Dark Theme Setup
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class WorkbenchApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Clinical Image Analysis & Enhancement Workbench")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        
        # State Variables
        self.pipeline = ImagePipeline()
        self.metadata_history = []
        self.zoom_scale = 1.0
        self.original_image_cache = None
        self.spectrum_mode = False
        self.notch_center = None
        self.roi_results = []
        self.template_box = None
        self.match_box = None
        self.drag_start = None
        self.drag_preview_id = None
        self.interaction_mode = ctk.StringVar(value="view")
        
        # UI Strings
        self.status = ctk.StringVar(value="Ready. Load an image to begin.")
        self.hist_mode = ctk.StringVar(value="Local")
        self.interp_method = ctk.StringVar(value="bilinear") 
        self.notch_type = ctk.StringVar(value="gaussian")
        self.notch_radius = ctk.StringVar(value="10")
        self.notch_order = ctk.StringVar(value="2")
        self.noise_type = ctk.StringVar(value="Gaussian")
        self.noise_mean = ctk.StringVar(value="0")
        self.noise_variance = ctk.StringVar(value="100")
        self.threshold_value = ctk.IntVar(value=128)
        self.threshold_label = ctk.StringVar(value="Threshold: 128")
        self.morph_size = ctk.StringVar(value="3")
        self.morph_shape = ctk.StringVar(value="square")
        
        self._build_ui()

    def _build_ui(self):
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # 1. Custom Dark Menu Bar
        menu_frame = ctk.CTkFrame(self, height=35, corner_radius=0, fg_color="#0D0E12")
        menu_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        
        ctk.CTkButton(menu_frame, text="Load Image", width=80, fg_color="transparent", text_color="#D1D5DB", hover_color="#1F2937", command=self.load_image).pack(side="left", padx=10, pady=5)
        ctk.CTkButton(menu_frame, text="Save Processed", width=80, fg_color="transparent", text_color="#D1D5DB", hover_color="#1F2937", command=self.save_current).pack(side="left", padx=5, pady=5)
        
        ctk.CTkLabel(menu_frame, text=" | ", text_color="#4B5563").pack(side="left", padx=10)
        
        ctk.CTkButton(menu_frame, text="Fit View", width=80, fg_color="transparent", text_color="#D1D5DB", hover_color="#1F2937", command=self.fit_to_window).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(menu_frame, text="100% Size", width=80, fg_color="transparent", text_color="#D1D5DB", hover_color="#1F2937", command=self.actual_size).pack(side="left", padx=5, pady=5)

        ctk.CTkLabel(menu_frame, text=" | ", text_color="#4B5563").pack(side="left", padx=10)
        self.btn_undo_top = ctk.CTkButton(menu_frame, text="Undo", width=70, fg_color="#374151", hover_color="#4B5563", command=self.undo, state="disabled")
        self.btn_undo_top.pack(side="left", padx=5, pady=5)
        self.btn_reset_top = ctk.CTkButton(menu_frame, text="Reset", width=70, fg_color="#9e2a2b", hover_color="#7a2021", command=self.reset, state="disabled")
        self.btn_reset_top.pack(side="left", padx=5, pady=5)

        # 2. Top Header
        header_frame = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color="#181A20")
        header_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        ctk.CTkLabel(header_frame, text="Clinical Image Processing - Team 5", 
                     font=ctk.CTkFont(size=20, weight="bold"), text_color="#3B82F6").pack(side="left", padx=20, pady=15)
        ctk.CTkLabel(header_frame, text="DIP Team 5", 
                     text_color="gray").pack(side="right", padx=20)

        # 3. Horizontal Phase Tabs
        tabs_frame = ctk.CTkFrame(self, height=45, corner_radius=0, fg_color="#111318")
        tabs_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        
        self.btn_tab_phase1 = ctk.CTkButton(tabs_frame, text="Phase 2 - Spatial Processing", 
                                            font=ctk.CTkFont(weight="bold"), fg_color="#1F2937", 
                                            border_color="#3B82F6", border_width=2, text_color="white",
                                            command=self.show_phase1_tab)
        self.btn_tab_phase1.pack(side="left", padx=(20, 5), pady=5, fill="y")
        
        self.btn_tab_metadata = ctk.CTkButton(tabs_frame, text="Metadata + Pipeline", 
                                              font=ctk.CTkFont(weight="bold"), fg_color="transparent", 
                                              border_width=0, text_color="gray",
                                              command=self.show_metadata_tab)
        self.btn_tab_metadata.pack(side="left", padx=5, pady=5, fill="y")

        # 4. Main Body Containers
        self._build_sidebar()
        
        self.main_container = ctk.CTkFrame(self, fg_color="#111318", corner_radius=0)
        self.main_container.grid(row=3, column=1, sticky="nsew", padx=10, pady=10)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        self._build_viewers()
        self._build_metadata_view()
        
        self.show_phase1_tab()

        # 5. Status Bar
        status_bar = ctk.CTkLabel(self, textvariable=self.status, anchor="w", fg_color="#111318", corner_radius=0, padx=20)
        status_bar.grid(row=4, column=0, columnspan=2, sticky="ew", ipady=5)

    def _build_sidebar(self):
        sidebar = ctk.CTkScrollableFrame(self, width=320, corner_radius=0, fg_color="#181A20")
        sidebar.grid(row=3, column=0, sticky="nsew")

        # 1. PRIMARY ACTIONS
        action_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        action_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        ctk.CTkButton(action_frame, text="Load Image", font=ctk.CTkFont(weight="bold"), command=self.load_image).pack(fill="x", pady=5)
        ctk.CTkButton(action_frame, text="Save Processed Image", fg_color="transparent", border_width=1, command=self.save_current).pack(fill="x", pady=5)

        ctk.CTkFrame(sidebar, height=1, fg_color="#374151").pack(fill="x", padx=10, pady=10)

        # 2. IMAGE VIEW & ZOOM
        ctk.CTkLabel(sidebar, text="IMAGE VIEW & ZOOM", font=ctk.CTkFont(weight="bold", size=11), text_color="#3B82F6").pack(anchor="w", padx=10, pady=(5, 5))
        
        ctk.CTkOptionMenu(sidebar, values=["nearest", "bilinear"], variable=self.interp_method, fg_color="#374151", button_color="#4B5563").pack(fill="x", padx=10, pady=5)
        
        zoom_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        zoom_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(zoom_frame, text="Zoom In (+)", command=lambda: self.zoom(1.25)).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(zoom_frame, text="Zoom Out (-)", command=lambda: self.zoom(0.8)).pack(side="left", fill="x", expand=True, padx=(5, 0))

        ctk.CTkFrame(sidebar, height=1, fg_color="#374151").pack(fill="x", padx=10, pady=10)

        # 3. SPATIAL FILTERING ENGINE (SEPARATED LIKE THE SCREENSHOT)
        ctk.CTkLabel(sidebar, text="SPATIAL FILTERS", font=ctk.CTkFont(weight="bold", size=11), text_color="#3B82F6").pack(anchor="w", padx=10, pady=(5, 5))
        
        param_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        param_frame.pack(fill="x", padx=10, pady=5)
        
        self.kernel_size = ctk.StringVar(value="3")
        self.variance = ctk.StringVar(value="1.0")
        
        ctk.CTkLabel(param_frame, text="Kernel size:").grid(row=0, column=0, sticky="w", pady=2)
        ctk.CTkEntry(param_frame, textvariable=self.kernel_size, width=60, height=25).grid(row=0, column=1, padx=5, pady=2, sticky="e")
        
        ctk.CTkLabel(param_frame, text="Gaussian variance:").grid(row=1, column=0, sticky="w", pady=2)
        ctk.CTkEntry(param_frame, textvariable=self.variance, width=60, height=25).grid(row=1, column=1, padx=5, pady=2, sticky="e")
        
        # Dedicated Filter Buttons
        ctk.CTkButton(sidebar, text="Average Filter", fg_color="#374151", hover_color="#4B5563", command=self.apply_average).pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Gaussian Filter", fg_color="#374151", hover_color="#4B5563", command=self.apply_gaussian).pack(fill="x", padx=10, pady=2)
        
        # Edge Options
        edge_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        edge_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        self.edge_operator = ctk.StringVar(value="sobel")
        self.edge_direction = ctk.StringVar(value="horizontal")
        
        ctk.CTkLabel(edge_frame, text="Edge operator:").grid(row=0, column=0, sticky="w", pady=2)
        ctk.CTkOptionMenu(edge_frame, values=["sobel", "prewitt"], variable=self.edge_operator, width=120, height=25, fg_color="#374151", button_color="#4B5563").grid(row=0, column=1, padx=5, pady=2, sticky="e")
        
        ctk.CTkLabel(edge_frame, text="Edge output:").grid(row=1, column=0, sticky="w", pady=2)
        ctk.CTkOptionMenu(edge_frame, values=["horizontal", "vertical", "magnitude"], variable=self.edge_direction, width=120, height=25, fg_color="#374151", button_color="#4B5563").grid(row=1, column=1, padx=5, pady=2, sticky="e")
        
        ctk.CTkButton(sidebar, text="Edge Detection", fg_color="#374151", hover_color="#4B5563", command=self.apply_edge).pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Median Filter", fg_color="#374151", hover_color="#4B5563", command=self.apply_median).pack(fill="x", padx=10, pady=2)

        ctk.CTkFrame(sidebar, height=1, fg_color="#374151").pack(fill="x", padx=10, pady=10)

        # 4. LOCAL HISTOGRAM (Separated visually)
        ctk.CTkLabel(sidebar, text="LOCAL HISTOGRAM", font=ctk.CTkFont(weight="bold", size=11), text_color="#3B82F6").pack(anchor="w", padx=10, pady=(5, 5))
        
        # Kept the Global/Local Radio buttons for complete functionality
        radio_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        radio_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkRadioButton(radio_frame, text="Global", variable=self.hist_mode, value="Global").pack(side="left", padx=(0, 15))
        ctk.CTkRadioButton(radio_frame, text="Local", variable=self.hist_mode, value="Local").pack(side="left")
        
        self.block_size = ctk.StringVar(value="8")
        block_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        block_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(block_frame, text="Local block size:").pack(side="left")
        ctk.CTkEntry(block_frame, textvariable=self.block_size, width=60, height=25).pack(side="right")
        
        ctk.CTkButton(sidebar, text="Apply Equalization", fg_color="#374151", hover_color="#4B5563", command=self.apply_equalization).pack(fill="x", padx=10, pady=10)

        ctk.CTkFrame(sidebar, height=1, fg_color="#374151").pack(fill="x", padx=10, pady=10)

        # 5. GEOMETRIC TRANSFORMS
        ctk.CTkLabel(sidebar, text="GEOMETRIC TRANSFORMS", font=ctk.CTkFont(weight="bold", size=11), text_color="#3B82F6").pack(anchor="w", padx=10, pady=(5, 5))
        
        transform_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        transform_frame.pack(fill="x", padx=10, pady=2)
        
        self.angle = ctk.StringVar(value="15")
        ctk.CTkLabel(transform_frame, text="Angle (deg):").grid(row=0, column=0, sticky="w")
        ctk.CTkEntry(transform_frame, textvariable=self.angle, width=60, height=25).grid(row=0, column=1, padx=5, pady=2, sticky="e")
        ctk.CTkButton(sidebar, text="Rotate Image", fg_color="transparent", border_width=1, command=self.apply_rotation).pack(fill="x", padx=10, pady=5)

        shear_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        shear_frame.pack(fill="x", padx=10, pady=2)
        
        self.shear_x = ctk.StringVar(value="0.2")
        self.shear_y = ctk.StringVar(value="0.0")
        ctk.CTkLabel(shear_frame, text="Shear X:").grid(row=0, column=0, sticky="w")
        ctk.CTkEntry(shear_frame, textvariable=self.shear_x, width=60, height=25).grid(row=0, column=1, padx=5, pady=2, sticky="e")
        ctk.CTkLabel(shear_frame, text="Shear Y:").grid(row=1, column=0, sticky="w")
        ctk.CTkEntry(shear_frame, textvariable=self.shear_y, width=60, height=25).grid(row=1, column=1, padx=5, pady=2, sticky="e")
        ctk.CTkButton(sidebar, text="Shear Image", fg_color="transparent", border_width=1, command=self.apply_shear).pack(fill="x", padx=10, pady=5)

        ctk.CTkFrame(sidebar, height=1, fg_color="#374151").pack(fill="x", padx=10, pady=10)

        # 6. PHASE 2 FREQUENCY, NOISE, ROI
        ctk.CTkLabel(sidebar, text="PHASE 2 FREQUENCY DOMAIN", font=ctk.CTkFont(weight="bold", size=11), text_color="#3B82F6").pack(anchor="w", padx=10, pady=(5, 5))

        ctk.CTkOptionMenu(sidebar, values=["ideal", "butterworth", "gaussian"], variable=self.notch_type, fg_color="#374151", button_color="#4B5563").pack(fill="x", padx=10, pady=5)

        notch_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        notch_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(notch_frame, text="Radius:").grid(row=0, column=0, sticky="w", pady=2)
        ctk.CTkEntry(notch_frame, textvariable=self.notch_radius, width=60, height=25).grid(row=0, column=1, padx=5, pady=2, sticky="e")
        ctk.CTkLabel(notch_frame, text="Order:").grid(row=1, column=0, sticky="w", pady=2)
        ctk.CTkEntry(notch_frame, textvariable=self.notch_order, width=60, height=25).grid(row=1, column=1, padx=5, pady=2, sticky="e")

        ctk.CTkButton(sidebar, text="Show Spectrum / Pick Notch", fg_color="#374151", hover_color="#4B5563", command=self.show_fourier_spectrum).pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Apply Selected Notch", fg_color="#374151", hover_color="#4B5563", command=self.apply_selected_notch).pack(fill="x", padx=10, pady=2)

        ctk.CTkFrame(sidebar, height=1, fg_color="#374151").pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(sidebar, text="NOISE + ROI ANALYSIS", font=ctk.CTkFont(weight="bold", size=11), text_color="#3B82F6").pack(anchor="w", padx=10, pady=(5, 5))

        ctk.CTkOptionMenu(sidebar, values=["Gaussian", "Uniform"], variable=self.noise_type, fg_color="#374151", button_color="#4B5563").pack(fill="x", padx=10, pady=5)
        noise_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        noise_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(noise_frame, text="Mean / Low:").grid(row=0, column=0, sticky="w", pady=2)
        ctk.CTkEntry(noise_frame, textvariable=self.noise_mean, width=60, height=25).grid(row=0, column=1, padx=5, pady=2, sticky="e")
        ctk.CTkLabel(noise_frame, text="Var / High:").grid(row=1, column=0, sticky="w", pady=2)
        ctk.CTkEntry(noise_frame, textvariable=self.noise_variance, width=60, height=25).grid(row=1, column=1, padx=5, pady=2, sticky="e")

        ctk.CTkButton(sidebar, text="Inject Synthetic Noise", fg_color="#374151", hover_color="#4B5563", command=self.apply_noise).pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Draw ROI on Processed", fg_color="transparent", border_width=1, command=self.activate_roi_tool).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(sidebar, text="Crop Template", fg_color="transparent", border_width=1, command=self.activate_template_tool).pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Find Template", fg_color="transparent", border_width=1, command=self.apply_template_match).pack(fill="x", padx=10, pady=2)

        ctk.CTkFrame(sidebar, height=1, fg_color="#374151").pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(sidebar, text="BONUS MORPHOLOGY", font=ctk.CTkFont(weight="bold", size=11), text_color="#3B82F6").pack(anchor="w", padx=10, pady=(5, 5))

        ctk.CTkLabel(sidebar, textvariable=self.threshold_label, text_color="#D1D5DB").pack(anchor="w", padx=10, pady=(0, 2))
        ctk.CTkSlider(sidebar, from_=0, to=255, number_of_steps=255, variable=self.threshold_value, command=self._on_threshold_change).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(sidebar, text="Binarize Image", fg_color="#374151", hover_color="#4B5563", command=self.apply_threshold).pack(fill="x", padx=10, pady=2)

        morph_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        morph_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(morph_frame, text="SE size:").grid(row=0, column=0, sticky="w", pady=2)
        ctk.CTkEntry(morph_frame, textvariable=self.morph_size, width=60, height=25).grid(row=0, column=1, padx=5, pady=2, sticky="e")
        ctk.CTkLabel(morph_frame, text="SE shape:").grid(row=1, column=0, sticky="w", pady=2)
        ctk.CTkOptionMenu(morph_frame, values=["square", "cross"], variable=self.morph_shape, width=120, height=25, fg_color="#374151", button_color="#4B5563").grid(row=1, column=1, padx=5, pady=2, sticky="e")

        ctk.CTkButton(sidebar, text="Erosion", fg_color="#374151", hover_color="#4B5563", command=lambda: self.apply_morphology("erosion")).pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Dilation", fg_color="#374151", hover_color="#4B5563", command=lambda: self.apply_morphology("dilation")).pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Opening", fg_color="#374151", hover_color="#4B5563", command=lambda: self.apply_morphology("opening")).pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Closing", fg_color="#374151", hover_color="#4B5563", command=lambda: self.apply_morphology("closing")).pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Boundary Extraction", fg_color="transparent", border_width=1, command=lambda: self.apply_morphology("boundary")).pack(fill="x", padx=10, pady=(2, 10))

    def _build_viewers(self):
        self.viewers_frame = ctk.CTkFrame(self.main_container, fg_color="#111318", corner_radius=0)
        self.viewers_frame.grid(row=0, column=0, sticky="nsew")
        
        self.viewers_frame.grid_columnconfigure((0, 1), weight=1, uniform="col")
        self.viewers_frame.grid_rowconfigure(0, weight=5) # Images get the dominant workspace
        self.viewers_frame.grid_rowconfigure(1, weight=1) # Histograms stay visible but compact

        def make_panel(parent, row, col, title, color):
            frame = ctk.CTkFrame(parent, fg_color="#181A20", corner_radius=5, border_width=1, border_color="#374151")
            frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
            frame.grid_rowconfigure(1, weight=1)
            frame.grid_columnconfigure(0, weight=1)
            label = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(weight="bold", size=12), text_color=color)
            label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
            canvas = tk.Canvas(frame, bg="#000000", highlightthickness=0)
            canvas.grid(row=1, column=0, sticky="nsew", padx=2, pady=(0, 2))
            return label, canvas

        self.label_orig, self.canvas_orig = make_panel(self.viewers_frame, 0, 0, "ORIGINAL", "#3B82F6")
        self.label_proc, self.canvas_proc = make_panel(self.viewers_frame, 0, 1, "PROCESSED", "#10B981")
        self.label_hist_orig, self.hist_orig_canvas = make_panel(self.viewers_frame, 1, 0, "Original Histogram", "#A0AAB5")
        self.label_hist_proc, self.hist_proc_canvas = make_panel(self.viewers_frame, 1, 1, "Processed Histogram", "#A0AAB5")

        self.canvas_proc.bind("<ButtonPress-1>", self._on_processed_press)
        self.canvas_proc.bind("<B1-Motion>", self._on_processed_drag)
        self.canvas_proc.bind("<ButtonRelease-1>", self._on_processed_release)

    def _build_metadata_view(self):
        self.metadata_frame = ctk.CTkFrame(self.main_container, fg_color="#181A20", corner_radius=5, border_width=1, border_color="#374151")
        self.metadata_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.metadata_frame.grid_rowconfigure(1, weight=1)
        self.metadata_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(self.metadata_frame, text="IMAGE METADATA & PIPELINE HISTORY", font=ctk.CTkFont(weight="bold", size=14), text_color="#3B82F6").grid(row=0, column=0, sticky="w", padx=15, pady=10)
        
        self.metadata_log = ctk.CTkTextbox(self.metadata_frame, fg_color="#0D0E12", text_color="#D1D5DB", font=ctk.CTkFont(family="Consolas", size=13))
        self.metadata_log.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.metadata_log.insert("0.0", "No images loaded yet. Load an image to see its metadata here.")
        self.metadata_log.configure(state="disabled")

    # ==================== TAB SWITCHING LOGIC ====================

    def show_phase1_tab(self):
        self.btn_tab_phase1.configure(fg_color="#1F2937", border_color="#3B82F6", border_width=2, text_color="white")
        self.btn_tab_metadata.configure(fg_color="transparent", border_width=0, text_color="gray")
        self.viewers_frame.tkraise()

    def show_metadata_tab(self):
        self.btn_tab_metadata.configure(fg_color="#1F2937", border_color="#3B82F6", border_width=2, text_color="white")
        self.btn_tab_phase1.configure(fg_color="transparent", border_width=0, text_color="gray")
        self._update_metadata_log()
        self.metadata_frame.tkraise()

    def _update_metadata_log(self):
        self.metadata_log.configure(state="normal")
        self.metadata_log.delete("0.0", "end")

        self.metadata_log.insert("end", "PIPELINE VIEW\n")
        self.metadata_log.insert("end", "-" * 80 + "\n")
        if self.pipeline.current is None:
            self.metadata_log.insert("end", "  No active image pipeline. Load an image to begin.\n\n")
        else:
            self.metadata_log.insert("end", "  00. Original image\n")
            if self.pipeline.operation_names:
                for index, operation in enumerate(self.pipeline.operation_names, start=1):
                    self.metadata_log.insert("end", f"  {index:02d}. {operation}\n")
            else:
                self.metadata_log.insert("end", "  No operations applied yet.\n")
            self.metadata_log.insert("end", "\n\n")

        if self.roi_results:
            self.metadata_log.insert("end", "ROI STATISTICS\n")
            self.metadata_log.insert("end", "-" * 80 + "\n")
            for index, result in enumerate(reversed(self.roi_results[-5:]), start=1):
                left, top, right, bottom = result["box"]
                self.metadata_log.insert(
                    "end",
                    f"  ROI {index}: box=({left}, {top})-({right}, {bottom}) | "
                    f"mean={result['mean']:.3f} | variance={result['variance']:.3f}\n",
                )
            self.metadata_log.insert("end", "\n\n")

        self.metadata_log.insert("end", "METADATA HISTORY\n")
        self.metadata_log.insert("end", "=" * 80 + "\n")
        if not self.metadata_history:
            self.metadata_log.insert("end", "No images loaded yet. Load an image to see its metadata here.")
        else:
            for entry in reversed(self.metadata_history):
                self.metadata_log.insert("end", f"LOADED: {entry['filename']}   |   DATE: {entry['date']}\n")
                self.metadata_log.insert("end", "-" * 80 + "\n")
                for key, value in entry['data'].items():
                    self.metadata_log.insert("end", f"  {key:<22}: {value}\n")
                self.metadata_log.insert("end", "\n\n")
                
        self.metadata_log.configure(state="disabled")

    # ==================== CORE OPERATIONS ====================

    def require_image(self):
        if self.pipeline.current is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return None
        return self.pipeline.current

    def _read_kernel_size(self):
        try:
            kernel_size = int(self.kernel_size.get())
        except ValueError:
            messagebox.showerror("Wrong Kernel Size", "Kernel size must be an odd integer of 3 or larger.")
            return None

        if kernel_size < 3 or kernel_size % 2 == 0:
            messagebox.showerror("Wrong Kernel Size", "Kernel size must be odd and at least 3.")
            return None
        return kernel_size

    def _read_gaussian_variance(self):
        try:
            variance = float(self.variance.get())
        except ValueError:
            messagebox.showerror("Invalid Variance", "Gaussian variance must be a number.")
            return None

        if variance <= 0:
            messagebox.showerror("Negative Variance", "Gaussian variance must be positive.")
            return None
        return variance

    def _update_pipeline_buttons(self):
        has_image = self.pipeline.original is not None
        has_history = bool(self.pipeline.history)
        self.btn_undo_top.configure(state="normal" if has_history else "disabled")
        self.btn_reset_top.configure(state="normal" if has_image else "disabled")

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        try:
            image, metadata = load_image(path)
            self.pipeline.load(image)
            self.original_image_cache = image
            self.spectrum_mode = False
            self.notch_center = None
            self.roi_results = []
            self.template_box = None
            self.match_box = None
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.metadata_history.append({
                "filename": os.path.basename(path),
                "date": timestamp,
                "data": metadata
            })
            
            self.status.set(f"Successfully loaded {os.path.basename(path)}")
            self._update_pipeline_buttons()
            self.show_phase1_tab()
            self.fit_to_window()
            
        except Exception as exc:
            messagebox.showerror("Load Failed", str(exc))

    def save_current(self):
        image = self.require_image()
        if image is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if not path: return
        save_image(path, image)
        self.status.set(f"Saved to {os.path.basename(path)}")

    # Specific Dedicated Filter Functions restored for the individual buttons
    def apply_average(self):
        k = self._read_kernel_size()
        if k is None: return
        self.apply_operation(f"Average Filter (k={k})", lambda img: filters.average_filter(img, k))

    def apply_gaussian(self):
        k = self._read_kernel_size()
        if k is None: return
        v = self._read_gaussian_variance()
        if v is None: return
        self.apply_operation(f"Gaussian Filter (k={k}, v={v})", lambda img: filters.gaussian_filter(img, k, v))

    def apply_median(self):
        k = self._read_kernel_size()
        if k is None: return
        self.apply_operation(f"Median Filter (k={k})", lambda img: filters.median_filter(img, k))

    def apply_edge(self):
        op = self.edge_operator.get()
        direction = self.edge_direction.get()
        self.apply_operation(f"{op.title()} ({direction})", lambda img: filters.edge_filter(img, op, direction))

    def apply_equalization(self):
        if self.hist_mode.get() == "Local":
            try:
                bs = int(self.block_size.get())
            except ValueError: return messagebox.showerror("Invalid Input", "Block size must be an integer.")
            self.apply_operation("Local Equalization", lambda img: local_histogram_equalization(img, bs))
        else:
            self.apply_operation("Global Equalization", lambda img: global_histogram_equalization(img))

    def apply_rotation(self):
        try:
            angle = float(self.angle.get())
            self.apply_operation(f"Rotation ({angle} deg)", lambda img: rotate(img, angle))
        except ValueError:
            messagebox.showerror("Invalid Input", "Angle must be a number.")

    def apply_shear(self):
        try:
            sx = float(self.shear_x.get())
            sy = float(self.shear_y.get())
            self.apply_operation(f"Shear (X:{sx}, Y:{sy})", lambda img: shear(img, sx, sy))
        except ValueError:
            messagebox.showerror("Invalid Input", "Shear values must be numbers.")

    def show_fourier_spectrum(self):
        image = self.require_image()
        if image is None: return
        spectrum_image, _ = frequency.fourier_spectrum(image)
        self.spectrum_mode = True
        self.notch_center = None
        self.interaction_mode.set("notch")
        self.photo_proc = ImageTk.PhotoImage(Image.fromarray(spectrum_image))
        self.label_proc.configure(text="FOURIER SPECTRUM - CLICK NOISE SPIKE")
        self._center_image_on_canvas(self.canvas_proc, self.photo_proc, spectrum_image.shape)
        self.status.set("Spectrum mode: click a bright periodic-noise spike in the processed panel.")

    def apply_selected_notch(self):
        if self.notch_center is None:
            messagebox.showwarning("No Notch Selected", "Show the Fourier spectrum, then click a bright noise spike first.")
            return
        try:
            radius = float(self.notch_radius.get())
            order = int(self.notch_order.get())
        except ValueError:
            messagebox.showerror("Invalid Notch Parameters", "Radius must be a number and order must be an integer.")
            return
        if radius <= 0:
            messagebox.showerror("Invalid Notch Radius", "Notch radius must be positive.")
            return

        cy, cx = self.notch_center
        kind = self.notch_type.get()
        self.spectrum_mode = False
        self.interaction_mode.set("view")
        self.apply_operation(
            f"{kind.title()} Notch Reject (u={cx}, v={cy}, r={radius:g})",
            lambda img: frequency.apply_notch_reject(img, (cy, cx), radius, kind, order),
        )

    def apply_noise(self):
        if self.noise_type.get() == "Gaussian":
            try:
                mean = float(self.noise_mean.get())
                variance = float(self.noise_variance.get())
            except ValueError:
                messagebox.showerror("Invalid Noise Parameters", "Gaussian mean and variance must be numbers.")
                return
            if variance < 0:
                messagebox.showerror("Negative Variance", "Gaussian noise variance must not be negative.")
                return
            self.apply_operation(f"Gaussian Noise (mean={mean:g}, var={variance:g})", lambda img: noise.add_gaussian_noise(img, mean, variance))
        else:
            try:
                low = float(self.noise_mean.get())
                high = float(self.noise_variance.get())
            except ValueError:
                messagebox.showerror("Invalid Noise Parameters", "Uniform low and high values must be numbers.")
                return
            if high <= low:
                messagebox.showerror("Invalid Uniform Range", "Uniform high value must be greater than low value.")
                return
            self.apply_operation(f"Uniform Noise ({low:g} to {high:g})", lambda img: noise.add_uniform_noise(img, low, high))

    def activate_roi_tool(self):
        if self.require_image() is None: return
        self.spectrum_mode = False
        self.interaction_mode.set("roi")
        self.render_all()
        self.status.set("ROI mode: drag a rectangle on the processed image.")

    def activate_template_tool(self):
        if self.require_image() is None: return
        self.spectrum_mode = False
        self.interaction_mode.set("template")
        self.render_all()
        self.status.set("Template mode: drag a rectangle around the template on the processed image.")

    def apply_template_match(self):
        image = self.require_image()
        if image is None: return
        if self.template_box is None:
            messagebox.showwarning("No Template", "Crop a template first by dragging on the processed image.")
            return

        x0, y0, x1, y1 = self.template_box
        template = image[y0:y1, x0:x1]

        try:
            y, x, corr = frequency.frequency_cross_correlation(image, template)
        except ValueError as exc:
            messagebox.showerror("Template Match Failed", str(exc))
            return

        h, w = template.shape[:2]
        self.match_box = (x, y, x + w, y + h)
        self._draw_box_on_processed(self.match_box, "#10B981", "match_box")
        self._draw_histogram(self.hist_proc_canvas, self._calculate_histogram(corr), color="#F59E0B")
        self.status.set(f"Template match found at x={x}, y={y}.")

    def _on_threshold_change(self, value):
        self.threshold_label.set(f"Threshold: {int(float(value))}")

    def _read_morphology_size(self):
        try:
            size = int(self.morph_size.get())
        except ValueError:
            messagebox.showerror("Wrong Structuring Element", "Structuring element size must be an odd integer of 3 or larger.")
            return None

        if size < 3 or size % 2 == 0:
            messagebox.showerror("Wrong Structuring Element", "Structuring element size must be odd and at least 3.")
            return None
        return size

    def apply_threshold(self):
        threshold = int(self.threshold_value.get())
        self.apply_operation(f"Binary Threshold (t={threshold})", lambda img: morphology.threshold_binary(img, threshold))

    def apply_morphology(self, operation):
        size = self._read_morphology_size()
        if size is None: return

        shape = self.morph_shape.get()
        operations = {
            "erosion": ("Erosion", morphology.erode),
            "dilation": ("Dilation", morphology.dilate),
            "opening": ("Opening", morphology.opening),
            "closing": ("Closing", morphology.closing),
            "boundary": ("Boundary Extraction", morphology.boundary_extraction),
        }
        title, func = operations[operation]
        self.apply_operation(f"{title} ({shape}, {size}x{size})", lambda img: func(img, size, shape))

    def apply_operation(self, name, func):
        image = self.require_image()
        if image is None: return
        
        self.status.set(f"Applying {name}... Please wait.")
        self.update_idletasks()

        def worker():
            try:
                result = func(image)
                self.after(0, self._on_operation_complete, result, name)
            except Exception as exc:
                self.after(0, self._on_operation_error, exc)

        threading.Thread(target=worker, daemon=True).start()

    def _on_operation_error(self, exc):
        message = str(exc)
        self.status.set(f"Error: {message}")
        messagebox.showerror("Operation Failed", message)

    def _on_operation_complete(self, result, name):
        self.spectrum_mode = False
        self.notch_center = None
        self.pipeline.apply(result, name)
        self.render_all()
        self._update_pipeline_buttons()
        self._update_metadata_log()
        self.status.set(f"Successfully applied: {name}")

    def undo(self):
        if self.pipeline.undo():
            self.render_all()
            self._update_pipeline_buttons()
            self._update_metadata_log()
            self.status.set("Undid last operation.")

    def reset(self):
        if self.pipeline.reset():
            self.spectrum_mode = False
            self.notch_center = None
            self.template_box = None
            self.match_box = None
            self.render_all()
            self._update_pipeline_buttons()
            self._update_metadata_log()
            self.status.set("Reset image to original state.")

    # ==================== PHASE 2 INTERACTIVE TOOLS ====================

    def _canvas_to_image_coords(self, canvas, event):
        box = getattr(canvas, "_image_box", None)
        source_shape = getattr(canvas, "_image_source_shape", None)
        if box is None or source_shape is None:
            return None

        left, top, right, bottom = box
        if event.x < left or event.x >= right or event.y < top or event.y >= bottom:
            return None

        src_h, src_w = source_shape[:2]
        x = int((event.x - left) * src_w / max(1, right - left))
        y = int((event.y - top) * src_h / max(1, bottom - top))
        return min(max(x, 0), src_w - 1), min(max(y, 0), src_h - 1)

    def _image_to_canvas_coords(self, canvas, x, y):
        box = getattr(canvas, "_image_box", None)
        source_shape = getattr(canvas, "_image_source_shape", None)
        if box is None or source_shape is None:
            return None

        left, top, right, bottom = box
        src_h, src_w = source_shape[:2]
        cx = left + x * (right - left) / max(1, src_w)
        cy = top + y * (bottom - top) / max(1, src_h)
        return cx, cy

    def _draw_box_on_processed(self, box, color, tag):
        coords0 = self._image_to_canvas_coords(self.canvas_proc, box[0], box[1])
        coords1 = self._image_to_canvas_coords(self.canvas_proc, box[2], box[3])
        if coords0 is None or coords1 is None:
            return
        self.canvas_proc.delete(tag)
        self.canvas_proc.create_rectangle(*coords0, *coords1, outline=color, width=2, tags=tag)

    def _on_processed_press(self, event):
        coords = self._canvas_to_image_coords(self.canvas_proc, event)
        if coords is None:
            return

        mode = self.interaction_mode.get()
        if mode == "notch" and self.spectrum_mode:
            x, y = coords
            self.notch_center = (y, x)
            self.canvas_proc.delete("notch_marker")
            marker = self._image_to_canvas_coords(self.canvas_proc, x, y)
            if marker is not None:
                mx, my = marker
                self.canvas_proc.create_oval(mx - 6, my - 6, mx + 6, my + 6, outline="#EF4444", width=2, tags="notch_marker")
                self.canvas_proc.create_line(mx - 10, my, mx + 10, my, fill="#EF4444", width=2, tags="notch_marker")
                self.canvas_proc.create_line(mx, my - 10, mx, my + 10, fill="#EF4444", width=2, tags="notch_marker")
            self.status.set(f"Selected notch at u={x}, v={y}. The conjugate notch will be mirrored automatically.")
            return

        if mode in {"roi", "template"}:
            self.drag_start = coords
            self.canvas_proc.delete("drag_preview")

    def _on_processed_drag(self, event):
        if self.drag_start is None or self.interaction_mode.get() not in {"roi", "template"}:
            return
        coords = self._canvas_to_image_coords(self.canvas_proc, event)
        if coords is None:
            return

        start_canvas = self._image_to_canvas_coords(self.canvas_proc, *self.drag_start)
        end_canvas = self._image_to_canvas_coords(self.canvas_proc, *coords)
        if start_canvas is None or end_canvas is None:
            return
        self.canvas_proc.delete("drag_preview")
        self.canvas_proc.create_rectangle(*start_canvas, *end_canvas, outline="#F59E0B", width=2, tags="drag_preview")

    def _on_processed_release(self, event):
        mode = self.interaction_mode.get()
        if self.drag_start is None or mode not in {"roi", "template"}:
            return

        coords = self._canvas_to_image_coords(self.canvas_proc, event)
        if coords is None:
            self.drag_start = None
            self.canvas_proc.delete("drag_preview")
            return

        x0, y0 = self.drag_start
        x1, y1 = coords
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if right - left < 2 or bottom - top < 2:
            messagebox.showerror("Invalid Selection", "Please drag a larger rectangle.")
            self.drag_start = None
            self.canvas_proc.delete("drag_preview")
            return

        if mode == "roi":
            self._complete_roi((left, top, right, bottom))
        else:
            self.template_box = (left, top, right, bottom)
            self._draw_box_on_processed(self.template_box, "#F59E0B", "template_box")
            self.status.set(f"Template cropped: {right - left} x {bottom - top}. Press Find Template.")

        self.drag_start = None

    def _complete_roi(self, box):
        image = self.require_image()
        if image is None:
            return
        try:
            result = roi.roi_statistics(image, box)
        except ValueError as exc:
            messagebox.showerror("ROI Failed", str(exc))
            return

        self.roi_results.append(result)
        self._draw_box_on_processed(result["box"], "#F59E0B", "roi_box")
        self._draw_histogram(self.hist_proc_canvas, result["histogram"], color="#F59E0B")
        self.label_hist_proc.configure(text="ROI Histogram")
        self._update_metadata_log()
        messagebox.showinfo(
            "ROI Statistics",
            f"Mean: {result['mean']:.3f}\nVariance: {result['variance']:.3f}\nSize: {result['width']} x {result['height']}",
        )
        self.status.set(f"ROI mean={result['mean']:.3f}, variance={result['variance']:.3f}.")

    # ==================== DUAL RENDERING & ZOOM ====================

    def zoom(self, factor):
        if self.require_image() is None: return
        self.zoom_scale = max(0.1, min(8.0, self.zoom_scale * factor))
        self.render_all()

    def fit_to_window(self):
        if self.require_image() is None: return
        self.update_idletasks()
        
        canvas_w = max(1, self.canvas_proc.winfo_width())
        canvas_h = max(1, self.canvas_proc.winfo_height())
        img_h, img_w = self.pipeline.current.shape[:2]
        
        self.zoom_scale = max(0.1, min(8.0, min(canvas_w / img_w, canvas_h / img_h)))
        self.render_all()

    def actual_size(self):
        if self.require_image() is None: return
        self.zoom_scale = 1.0
        self.render_all()

    def render_all(self):
        proc_image = self.require_image()
        orig_image = self.original_image_cache
        if proc_image is None or orig_image is None: return
        
        self.status.set(f"Rendering views at {self.zoom_scale:.2f}x using {self.interp_method.get().title()} interpolation...")
        self.update_idletasks()

        def worker():
            try:
                disp_orig = display_ready(orig_image)
                disp_proc = display_ready(proc_image)

                if self.zoom_scale != 1.0:
                    method = self.interp_method.get()
                    disp_orig = resize(disp_orig, self.zoom_scale, method)
                    disp_proc = resize(disp_proc, self.zoom_scale, method)

                hist_orig_data = self._calculate_histogram(orig_image)
                hist_proc_data = self._calculate_histogram(proc_image)

                self.after(0, self._update_ui_canvases, disp_orig, disp_proc, hist_orig_data, hist_proc_data)
            except Exception as exc:
                self.after(0, lambda: self.status.set(f"Render failed: {str(exc)}"))

        threading.Thread(target=worker, daemon=True).start()

    def _update_ui_canvases(self, disp_orig, disp_proc, hist_orig_data, hist_proc_data):
        self.photo_orig = ImageTk.PhotoImage(Image.fromarray(disp_orig))
        self.photo_proc = ImageTk.PhotoImage(Image.fromarray(disp_proc))
        
        self.label_proc.configure(text="PROCESSED")
        self.label_hist_proc.configure(text="Processed Histogram")
        self._center_image_on_canvas(self.canvas_orig, self.photo_orig, self.original_image_cache.shape)
        self._center_image_on_canvas(self.canvas_proc, self.photo_proc, self.pipeline.current.shape)

        self._draw_histogram(self.hist_orig_canvas, hist_orig_data, color="#3B82F6")
        self._draw_histogram(self.hist_proc_canvas, hist_proc_data, color="#10B981")

        if self.template_box is not None:
            self._draw_box_on_processed(self.template_box, "#F59E0B", "template_box")
        if self.match_box is not None:
            self._draw_box_on_processed(self.match_box, "#10B981", "match_box")
        
        self.status.set(f"Ready | Zoom: {self.zoom_scale:.2f}x | Interp: {self.interp_method.get().title()}")

    def _center_image_on_canvas(self, canvas, photo, source_shape=None):
        canvas.delete("all")
        self.update_idletasks()
        x_center = max(1, canvas.winfo_width()) // 2
        y_center = max(1, canvas.winfo_height()) // 2
        canvas.create_image(x_center, y_center, image=photo, anchor="center")
        half_w = photo.width() / 2
        half_h = photo.height() / 2
        canvas._image_box = (x_center - half_w, y_center - half_h, x_center + half_w, y_center + half_h)
        canvas._image_source_shape = source_shape

    def _calculate_histogram(self, image):
        if len(image.shape) == 3:
            gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        else:
            gray = image
        counts, _ = np.histogram(gray.flatten(), 256, [0, 256])
        return counts

    def _draw_histogram(self, canvas, hist_data, color):
        """Draws a clinical histogram with background-noise clipping."""
        canvas.delete("all")
        self.update_idletasks()
        
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 10 or h <= 10: return

        # --- THE FIX: Clinical Clipping ---
        # We ignore the massive spikes at 0 (black) and 255 (white) 
        # to find a better scale for the actual tissue data.
        mid_data = hist_data[1:254] 
        if mid_data.size > 0 and mid_data.max() > 0:
            # Set the max height to 1.5x the average of tissue pixels
            # instead of the absolute max (which is just background)
            max_val = np.percentile(mid_data, 95) * 1.5
        else:
            max_val = hist_data.max()
            
        if max_val == 0: max_val = 1
        
        bar_width = w / 256
        for i in range(256):
            # Clip the bar height so it doesn't go off the top of the panel
            raw_h = (hist_data[i] / max_val) * (h * 0.8)
            bar_h = min(raw_h, h * 0.9) 
            
            x0 = i * bar_width
            y0 = h
            x1 = x0 + bar_width
            y1 = h - bar_h
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)

if __name__ == "__main__":
    app = WorkbenchApp()
    app.mainloop()
