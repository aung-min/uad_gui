from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from src.data.mvtec_ad_dataset import VALID_MVTEC_AD_CATEGORIES
from src.inference.ae_inference_engine import (
    AEInferenceEngine,
    export_ae_inference_result,
)
from src.inference.feature_inference_engine import (
    FeatureInferenceEngine,
    export_inference_result,
    load_rgb_image,
)
from src.inference.hybrid_inference_engine import (
    HybridInferenceEngine,
    export_hybrid_inference_result,
)
from src.utils.anomaly_bboxes import (
    draw_anomaly_contours,
    extract_anomaly_contours,
)


DISPLAY_IMAGE_SIZE = (360, 360)
DASHBOARD_IMAGE_SIZE = (900, 520)
COMPARE_IMAGE_SIZE = (760, 440)
SECTION5_CHART_SIZE = (760, 440)
MODEL_NAMES = ("feature", "ae", "hybrid")

GUI_FONT_FAMILY = "Segoe UI"
GUI_FONT_SIZE = 12
GUI_LABELFRAME_TITLE_SIZE = 12
GUI_STATUS_FONT_SIZE = 13
GUI_FIXED_FONT_FAMILY = "Consolas"
GUI_FIXED_FONT_SIZE = 11
GUI_TREE_ROWHEIGHT = 28

DEFAULT_STARTUP_RUN_DIR = Path("data") / "mvtec_ad" / "bottle" / "test" / "contamination"

def resolve_startup_run_dir() -> str:
    if DEFAULT_STARTUP_RUN_DIR.is_dir():
        return str(DEFAULT_STARTUP_RUN_DIR.resolve())
    return ""


def configure_app_style(root: tk.Misc) -> None:
    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(family=GUI_FONT_FAMILY, size=GUI_FONT_SIZE)

    text_font = tkfont.nametofont("TkTextFont")
    text_font.configure(family=GUI_FONT_FAMILY, size=GUI_FONT_SIZE)

    fixed_font = tkfont.nametofont("TkFixedFont")
    fixed_font.configure(family=GUI_FIXED_FONT_FAMILY, size=GUI_FIXED_FONT_SIZE)

    root.option_add("*Font", default_font)

    style = ttk.Style(root)
    style.configure(".", font=(GUI_FONT_FAMILY, GUI_FONT_SIZE))
    style.configure("TLabel", font=(GUI_FONT_FAMILY, GUI_FONT_SIZE))
    style.configure("TButton", font=(GUI_FONT_FAMILY, GUI_FONT_SIZE))
    style.configure("TEntry", font=(GUI_FONT_FAMILY, GUI_FONT_SIZE))
    style.configure("TCombobox", font=(GUI_FONT_FAMILY, GUI_FONT_SIZE))
    style.configure("TCheckbutton", font=(GUI_FONT_FAMILY, GUI_FONT_SIZE))
    style.configure("TRadiobutton", font=(GUI_FONT_FAMILY, GUI_FONT_SIZE))
    style.configure("TLabelframe", font=(GUI_FONT_FAMILY, GUI_FONT_SIZE))
    style.configure("TLabelframe.Label", font=(GUI_FONT_FAMILY, GUI_LABELFRAME_TITLE_SIZE, "bold"))
    style.configure("TNotebook.Tab", font=(GUI_FONT_FAMILY, GUI_FONT_SIZE))
    style.configure("Treeview", font=(GUI_FONT_FAMILY, GUI_FONT_SIZE), rowheight=GUI_TREE_ROWHEIGHT)
    style.configure("Treeview.Heading", font=(GUI_FONT_FAMILY, GUI_FONT_SIZE, "bold"))


def resize_for_display(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    image = image.copy()
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def anomaly_map_to_heat_image(anomaly_map: np.ndarray) -> Image.Image:
    anomaly_map = anomaly_map.astype(np.float32)
    min_v = float(anomaly_map.min())
    max_v = float(anomaly_map.max())

    if max_v - min_v < 1e-12:
        norm = np.zeros_like(anomaly_map, dtype=np.float32)
    else:
        norm = (anomaly_map - min_v) / (max_v - min_v)

    heat = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(heat, mode="L").convert("RGB")


def load_json_if_exists(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class ZoomImageWindow:
    def __init__(self, parent: tk.Tk) -> None:
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Zoomable Graph")
        self.window.geometry("1280x920")

        configure_app_style(self.window)

        self.original_image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.zoom_factor = 1.5

        self._build_layout()

    def _build_layout(self) -> None:
        main = ttk.Frame(self.window, padding=8)
        main.pack(fill="both", expand=True)

        toolbar = ttk.Frame(main)
        toolbar.pack(fill="x", pady=(0, 8))

        ttk.Button(toolbar, text="Zoom In", command=self.zoom_in).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Zoom Out", command=self.zoom_out).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Fit", command=self.fit_to_window).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Reset", command=self.reset_zoom).pack(side="left", padx=4)

        self.zoom_label = ttk.Label(toolbar, text="Zoom: 100%")
        self.zoom_label.pack(side="left", padx=12)

        canvas_frame = ttk.Frame(main)
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, background="#202020", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        y_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        y_scroll.pack(side="right", fill="y")

        x_scroll = ttk.Scrollbar(main, orient="horizontal", command=self.canvas.xview)
        x_scroll.pack(fill="x")

        self.canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.window.bind("<Configure>", self._on_window_configure)

    def _on_mousewheel(self, event) -> None:
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def _on_window_configure(self, _event=None) -> None:
        if self.original_image is not None and self.zoom_factor == 1.0:
            self.render_image()

    def open_image(self, image: Image.Image, title: str) -> None:
        self.original_image = image.copy()
        self.window.title(title)
        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()
        self.fit_to_window()

    def fit_to_window(self) -> None:
        if self.original_image is None:
            return

        self.window.update_idletasks()
        canvas_w = max(200, self.canvas.winfo_width())
        canvas_h = max(200, self.canvas.winfo_height())
        img_w, img_h = self.original_image.size

        scale_w = canvas_w / max(1, img_w)
        scale_h = canvas_h / max(1, img_h)
        self.zoom_factor = max(0.05, min(scale_w, scale_h))
        self.render_image()

    def reset_zoom(self) -> None:
        self.zoom_factor = 1.0
        self.render_image()

    def zoom_in(self) -> None:
        self.zoom_factor *= 1.25
        self.render_image()

    def zoom_out(self) -> None:
        self.zoom_factor = max(0.05, self.zoom_factor / 1.25)
        self.render_image()

    def render_image(self) -> None:
        if self.original_image is None:
            return

        new_w = max(1, int(self.original_image.width * self.zoom_factor))
        new_h = max(1, int(self.original_image.height * self.zoom_factor))
        resized = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, new_w, new_h))
        self.zoom_label.configure(text=f"Zoom: {self.zoom_factor * 100:.0f}%")


class DashboardWindow:
    def __init__(self, parent: tk.Tk, category: str = "bottle", detector: str = "feature") -> None:
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Comparison and Evaluation Dashboards")
        self.window.geometry("1580x1020")

        configure_app_style(self.window)

        self.category_var = tk.StringVar(value=category)
        self.detector_var = tk.StringVar(value=detector)

        self.feature_reports_dir = Path("outputs/reports")
        self.ae_reports_dir = Path("outputs/ae_reports")
        self.hybrid_reports_dir = Path("outputs/hybrid_reports")
        self.model_thresholds_dir = Path("outputs/model_thresholds")
        self.dashboards_root_dir = Path("outputs/reports/dashboards")
        self.exploration_dir = Path("outputs/exploration")
        self.comparison_dashboards_dir = Path("outputs/exploration/dashboards")
        self.section5_outputs_dir = self.comparison_dashboards_dir

        self.eval_dashboard_photo: ImageTk.PhotoImage | None = None
        self.compare_left_photo: ImageTk.PhotoImage | None = None
        self.compare_right_photo: ImageTk.PhotoImage | None = None
        self.rq1_chart_photo: ImageTk.PhotoImage | None = None
        self.rq2_left_photo: ImageTk.PhotoImage | None = None
        self.rq2_right_photo: ImageTk.PhotoImage | None = None
        self.rq3_chart_photo: ImageTk.PhotoImage | None = None
        self.zoom_window: ZoomImageWindow | None = None

        self._build_layout()
        self.refresh_evaluation_tab()
        self.refresh_comparison_tab()
        self.refresh_section5_tab()

    def _build_layout(self) -> None:
        main = ttk.Frame(self.window, padding=12)
        main.pack(fill="both", expand=True)

        notebook = ttk.Notebook(main)
        notebook.pack(fill="both", expand=True)

        self.eval_tab = ttk.Frame(notebook)
        self.compare_tab = ttk.Frame(notebook)
        self.section5_tab = ttk.Frame(notebook)

        notebook.add(self.eval_tab, text="Evaluation Dashboard")
        notebook.add(self.compare_tab, text="Comparison Dashboard")
        notebook.add(self.section5_tab, text="Section 5 Results")

        self._build_eval_tab()
        self._build_compare_tab()
        self._build_section5_tab()

    def _create_zoomable_image_panel(
        self,
        parent: ttk.Frame,
        title: str,
        max_size: tuple[int, int],
    ) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        label = ttk.Label(frame, text="No image", anchor="center", cursor="hand2")
        label.pack(fill="both", expand=True)
        frame.image_label = label
        frame.max_size = max_size
        frame.source_image = None
        label.bind("<Button-1>", lambda _e, panel=frame, t=title: self.open_zoom_from_panel(panel, t))
        return frame

    def _create_table_panel(
        self,
        parent: ttk.Frame,
        title: str,
        height: int = 12,
    ) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=8)

        info_label = ttk.Label(frame, text="", anchor="w")
        info_label.pack(fill="x", pady=(0, 6))

        table_frame = ttk.Frame(frame)
        table_frame.pack(fill="both", expand=True)

        tree = ttk.Treeview(table_frame, show="headings", height=height)
        tree.pack(side="left", fill="both", expand=True)

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        y_scroll.pack(side="right", fill="y")
        tree.configure(yscrollcommand=y_scroll.set)

        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        x_scroll.pack(fill="x")
        tree.configure(xscrollcommand=x_scroll.set)

        frame.info_label = info_label
        frame.table_tree = tree
        return frame

    def _set_panel_image(
        self,
        panel: ttk.LabelFrame,
        image: Image.Image,
        attr_name: str,
    ) -> None:
        resized = resize_for_display(image, panel.max_size)
        photo = ImageTk.PhotoImage(resized)
        panel.image_label.configure(image=photo, text="")
        panel.image_label.image = photo
        panel.source_image = image.copy()
        setattr(self, attr_name, photo)

    def _clear_panel(self, panel: ttk.LabelFrame, text: str = "No image") -> None:
        panel.image_label.configure(image="", text=text)
        panel.image_label.image = None
        panel.source_image = None

    def _set_text(self, widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _clear_table_panel(self, panel: ttk.LabelFrame, text: str = "No table") -> None:
        tree = panel.table_tree
        for item in tree.get_children():
            tree.delete(item)
        tree["columns"] = ()
        panel.info_label.configure(text=text)

    def _set_table_from_dataframe(
        self,
        panel: ttk.LabelFrame,
        df: pd.DataFrame,
        source_path: Path | None = None,
    ) -> None:
        tree = panel.table_tree

        for item in tree.get_children():
            tree.delete(item)

        columns = [str(c) for c in df.columns.tolist()]
        tree["columns"] = columns

        for col in columns:
            tree.heading(col, text=col)

            series = df[col].astype(str) if len(df) > 0 else pd.Series([col])
            max_len = max([len(col)] + series.map(len).tolist())
            width = min(260, max(110, max_len * 9))
            tree.column(col, width=width, anchor="center")

        for _, row in df.iterrows():
            values = []
            for col in columns:
                value = row[col]
                if pd.isna(value):
                    values.append("")
                elif isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            tree.insert("", "end", values=values)

        info_text = f"Rows: {len(df)}"
        if source_path is not None:
            info_text += f"    Source: {source_path}"
        panel.info_label.configure(text=info_text)

    def open_zoom_from_panel(self, panel: ttk.LabelFrame, title: str) -> None:
        source_image = getattr(panel, "source_image", None)
        if source_image is None:
            return
        if self.zoom_window is None or not self.zoom_window.window.winfo_exists():
            self.zoom_window = ZoomImageWindow(self.window)
        self.zoom_window.open_image(source_image, title)

    def _build_eval_tab(self) -> None:
        top = ttk.LabelFrame(self.eval_tab, text="Evaluation Controls", padding=10)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Category").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        cat_combo = ttk.Combobox(
            top,
            textvariable=self.category_var,
            values=list(VALID_MVTEC_AD_CATEGORIES),
            state="readonly",
            width=18,
        )
        cat_combo.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        cat_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_evaluation_tab())

        ttk.Label(top, text="Detector").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        det_combo = ttk.Combobox(
            top,
            textvariable=self.detector_var,
            values=list(MODEL_NAMES),
            state="readonly",
            width=12,
        )
        det_combo.grid(row=0, column=3, sticky="w", padx=6, pady=6)
        det_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_evaluation_tab())

        ttk.Button(top, text="Refresh", command=self.refresh_evaluation_tab).grid(
            row=0,
            column=4,
            sticky="ew",
            padx=6,
            pady=6,
        )

        body = ttk.Frame(self.eval_tab)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        self.eval_dashboard_panel = self._create_zoomable_image_panel(
            body,
            "Saved Evaluation Dashboard (click to zoom)",
            DASHBOARD_IMAGE_SIZE,
        )
        self.eval_dashboard_panel.pack(fill="both", expand=True, pady=(0, 8))

        text_frame = ttk.LabelFrame(body, text="Full Detail Information", padding=8)
        text_frame.pack(fill="both", expand=True)

        self.eval_text = tk.Text(
            text_frame,
            wrap="word",
            height=20,
            font=(GUI_FIXED_FONT_FAMILY, GUI_FIXED_FONT_SIZE),
        )
        self.eval_text.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.eval_text.yview)
        scroll.pack(side="right", fill="y")
        self.eval_text.configure(yscrollcommand=scroll.set)
        self.eval_text.configure(state="disabled")

    def _build_compare_tab(self) -> None:
        top = ttk.LabelFrame(self.compare_tab, text="Comparison Controls", padding=10)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Category").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        cat_combo = ttk.Combobox(
            top,
            textvariable=self.category_var,
            values=list(VALID_MVTEC_AD_CATEGORIES),
            state="readonly",
            width=18,
        )
        cat_combo.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        cat_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_comparison_tab())

        ttk.Button(top, text="Refresh", command=self.refresh_comparison_tab).grid(
            row=0,
            column=2,
            sticky="ew",
            padx=6,
            pady=6,
        )

        images_frame = ttk.Frame(self.compare_tab)
        images_frame.pack(fill="both", expand=False, padx=8, pady=8)

        self.compare_left_panel = self._create_zoomable_image_panel(
            images_frame,
            "Image-level Comparison (click to zoom)",
            COMPARE_IMAGE_SIZE,
        )
        self.compare_right_panel = self._create_zoomable_image_panel(
            images_frame,
            "Pixel-level Comparison (click to zoom)",
            COMPARE_IMAGE_SIZE,
        )

        self.compare_left_panel.grid(row=0, column=0, sticky="nsew", padx=6)
        self.compare_right_panel.grid(row=0, column=1, sticky="nsew", padx=6)

        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)

        text_frame = ttk.LabelFrame(self.compare_tab, text="Full Detail Information", padding=8)
        text_frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.compare_text = tk.Text(
            text_frame,
            wrap="word",
            height=22,
            font=(GUI_FIXED_FONT_FAMILY, GUI_FIXED_FONT_SIZE),
        )
        self.compare_text.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.compare_text.yview)
        scroll.pack(side="right", fill="y")
        self.compare_text.configure(yscrollcommand=scroll.set)
        self.compare_text.configure(state="disabled")

    def _build_section5_tab(self) -> None:
        main = ttk.Frame(self.section5_tab, padding=8)
        main.pack(fill="both", expand=True)

        header = ttk.LabelFrame(main, text="Section 5 Mapping", padding=10)
        header.pack(fill="x", pady=(0, 8))

        ttk.Label(
            header,
            text=(
                "RQ1: Image AUROC chart + best-detector summary table    |    "
                "RQ2: Image AUROC chart + Pixel AUROC chart    |    "
                "RQ3: Best F1 chart + threshold summary table"
            ),
        ).pack(anchor="w")

        sub_notebook = ttk.Notebook(main)
        sub_notebook.pack(fill="both", expand=True)

        self.rq1_tab = ttk.Frame(sub_notebook)
        self.rq2_tab = ttk.Frame(sub_notebook)
        self.rq3_tab = ttk.Frame(sub_notebook)

        sub_notebook.add(self.rq1_tab, text="RQ1")
        sub_notebook.add(self.rq2_tab, text="RQ2")
        sub_notebook.add(self.rq3_tab, text="RQ3")

        self._build_rq1_tab()
        self._build_rq2_tab()
        self._build_rq3_tab()

    def _build_rq1_tab(self) -> None:
        body = ttk.Frame(self.rq1_tab)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        self.rq1_chart_panel = self._create_zoomable_image_panel(
            body,
            "RQ1 Left: Image AUROC grouped chart",
            SECTION5_CHART_SIZE,
        )
        self.rq1_table_panel = self._create_table_panel(
            body,
            "RQ1 Right: Best-detector summary table",
            height=16,
        )

        self.rq1_chart_panel.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.rq1_table_panel.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

    def _build_rq2_tab(self) -> None:
        body = ttk.Frame(self.rq2_tab)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        self.rq2_left_panel = self._create_zoomable_image_panel(
            body,
            "RQ2 Left: Image AUROC chart",
            SECTION5_CHART_SIZE,
        )
        self.rq2_right_panel = self._create_zoomable_image_panel(
            body,
            "RQ2 Right: Pixel AUROC chart",
            SECTION5_CHART_SIZE,
        )

        self.rq2_left_panel.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.rq2_right_panel.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

    def _build_rq3_tab(self) -> None:
        body = ttk.Frame(self.rq3_tab)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        self.rq3_chart_panel = self._create_zoomable_image_panel(
            body,
            "RQ3 Left: Best F1 grouped chart",
            SECTION5_CHART_SIZE,
        )
        self.rq3_table_panel = self._create_table_panel(
            body,
            "RQ3 Right: Threshold summary table",
            height=16,
        )

        self.rq3_chart_panel.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.rq3_table_panel.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

    def evaluation_summary_json_path(self, detector: str, category: str) -> Path:
        detector = detector.strip().lower()
        if detector == "feature":
            return self.feature_reports_dir / category / f"{category}_evaluation_summary.json"
        if detector == "ae":
            return self.ae_reports_dir / category / f"{category}_ae_evaluation_summary.json"
        return self.hybrid_reports_dir / category / f"{category}_hybrid_evaluation_summary.json"

    def threshold_summary_json_path(self, detector: str, category: str) -> Path:
        detector = detector.strip().lower()
        json_path = self.model_thresholds_dir / detector / category / f"{category}_{detector}_thresholds.json"
        if detector == "feature" and not json_path.exists():
            legacy_path = Path("outputs/reports/thresholds") / category / f"{category}_thresholds.json"
            if legacy_path.exists():
                return legacy_path
        return json_path

    def detector_dashboard_dir(self, detector: str, category: str) -> Path:
        return self.dashboards_root_dir / detector / category

    def dashboard_summary_json_path(self, detector: str, category: str) -> Path:
        detector = detector.strip().lower()
        path = self.detector_dashboard_dir(detector, category) / f"{category}_{detector}_dashboard_summary.json"
        if detector == "feature" and not path.exists():
            legacy_path = self.dashboards_root_dir / category / f"{category}_dashboard_summary.json"
            if legacy_path.exists():
                return legacy_path
        return path

    def dashboard_png_path(self, detector: str, category: str) -> Path:
        detector = detector.strip().lower()
        path = self.detector_dashboard_dir(detector, category) / f"{category}_{detector}_confusion_threshold_dashboard.png"
        if detector == "feature" and not path.exists():
            legacy_path = self.dashboards_root_dir / category / f"{category}_confusion_threshold_dashboard.png"
            if legacy_path.exists():
                return legacy_path
        return path

    def sync_selection(self, category: str, detector: str) -> None:
        self.category_var.set(category)
        self.detector_var.set(detector)
        self.refresh_evaluation_tab()
        self.refresh_comparison_tab()
        self.refresh_section5_tab()

    def refresh_evaluation_tab(self) -> None:
        category = self.category_var.get().strip()
        detector = self.detector_var.get().strip().lower()

        eval_json = self.evaluation_summary_json_path(detector, category)
        threshold_json = self.threshold_summary_json_path(detector, category)
        dashboard_json = self.dashboard_summary_json_path(detector, category)
        dashboard_png = self.dashboard_png_path(detector, category)

        eval_payload = load_json_if_exists(eval_json)
        threshold_payload = load_json_if_exists(threshold_json)
        dashboard_payload = load_json_if_exists(dashboard_json)

        details = {
            "category": category,
            "detector": detector,
            "evaluation_summary": eval_payload,
            "threshold_summary": threshold_payload,
            "dashboard_summary": dashboard_payload,
            "paths": {
                "evaluation_json": str(eval_json),
                "threshold_json": str(threshold_json),
                "dashboard_json": str(dashboard_json),
                "dashboard_png": str(dashboard_png),
            },
        }
        self._set_text(self.eval_text, json.dumps(details, indent=2))

        if dashboard_png.exists():
            self._set_panel_image(
                self.eval_dashboard_panel,
                Image.open(dashboard_png).convert("RGB"),
                "eval_dashboard_photo",
            )
        else:
            self._clear_panel(self.eval_dashboard_panel, text="No saved dashboard image for this detector")

    def refresh_comparison_tab(self) -> None:
        category = self.category_var.get().strip()

        left_png = self.comparison_dashboards_dir / "image_auroc_comparison.png"
        right_png = self.comparison_dashboards_dir / "pixel_auroc_comparison.png"

        if left_png.exists():
            self._set_panel_image(
                self.compare_left_panel,
                Image.open(left_png).convert("RGB"),
                "compare_left_photo",
            )
        else:
            self._clear_panel(self.compare_left_panel, text="Image-level comparison chart not found")

        if right_png.exists():
            self._set_panel_image(
                self.compare_right_panel,
                Image.open(right_png).convert("RGB"),
                "compare_right_photo",
            )
        else:
            self._clear_panel(self.compare_right_panel, text="Pixel-level comparison chart not found")

        category_summary_json = self.exploration_dir / category / f"{category}_broader_exploration_summary.json"
        global_summary_json = self.comparison_dashboards_dir / "model_comparison_summary.json"

        details = {
            "category": category,
            "category_broader_exploration_summary": load_json_if_exists(category_summary_json),
            "global_model_comparison_summary": load_json_if_exists(global_summary_json),
            "paths": {
                "category_summary_json": str(category_summary_json),
                "global_summary_json": str(global_summary_json),
                "image_comparison_png": str(left_png),
                "pixel_comparison_png": str(right_png),
            },
        }
        self._set_text(self.compare_text, json.dumps(details, indent=2))

    def refresh_section5_tab(self) -> None:
        rq1_chart_png = self.section5_outputs_dir / "image_auroc_comparison.png"
        rq1_table_csv = self.section5_outputs_dir / "best_image_detector_by_category.csv"

        rq2_left_png = self.section5_outputs_dir / "image_auroc_comparison.png"
        rq2_right_png = self.section5_outputs_dir / "pixel_auroc_comparison.png"

        rq3_chart_png = self.section5_outputs_dir / "best_f1_comparison.png"
        rq3_table_csv = self.section5_outputs_dir / "threshold_summary_table.csv"

        if rq1_chart_png.exists():
            self._set_panel_image(
                self.rq1_chart_panel,
                Image.open(rq1_chart_png).convert("RGB"),
                "rq1_chart_photo",
            )
        else:
            self._clear_panel(self.rq1_chart_panel, text="RQ1 Image AUROC chart not found")

        if rq1_table_csv.exists():
            self._set_table_from_dataframe(
                self.rq1_table_panel,
                pd.read_csv(rq1_table_csv),
                rq1_table_csv,
            )
        else:
            self._clear_table_panel(self.rq1_table_panel, text="RQ1 summary table CSV not found")

        if rq2_left_png.exists():
            self._set_panel_image(
                self.rq2_left_panel,
                Image.open(rq2_left_png).convert("RGB"),
                "rq2_left_photo",
            )
        else:
            self._clear_panel(self.rq2_left_panel, text="RQ2 Image AUROC chart not found")

        if rq2_right_png.exists():
            self._set_panel_image(
                self.rq2_right_panel,
                Image.open(rq2_right_png).convert("RGB"),
                "rq2_right_photo",
            )
        else:
            self._clear_panel(self.rq2_right_panel, text="RQ2 Pixel AUROC chart not found")

        if rq3_chart_png.exists():
            self._set_panel_image(
                self.rq3_chart_panel,
                Image.open(rq3_chart_png).convert("RGB"),
                "rq3_chart_photo",
            )
        else:
            self._clear_panel(self.rq3_chart_panel, text="RQ3 Best F1 chart not found")

        if rq3_table_csv.exists():
            self._set_table_from_dataframe(
                self.rq3_table_panel,
                pd.read_csv(rq3_table_csv),
                rq3_table_csv,
            )
        else:
            self._clear_table_panel(self.rq3_table_panel, text="RQ3 threshold summary CSV not found")


class AnomalyTkinterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MVTec AD Anomaly Detector")
        self.root.geometry("1600x1040")
        self.root.minsize(1500, 980)

        configure_app_style(self.root)

        self.category_var = tk.StringVar(value="bottle")
        self.model_var = tk.StringVar(value="feature")

        self.feature_memory_bank_var = tk.StringVar(value=self.default_feature_memory_bank_path("bottle"))
        self.ae_checkpoint_var = tk.StringVar(value=self.default_ae_checkpoint_path("bottle"))
        self.thresholds_root_var = tk.StringVar(value="outputs/model_thresholds")

        self.image_path_var = tk.StringVar(value="")
        self.image_dir_var = tk.StringVar(value=resolve_startup_run_dir())
        self.output_dir_var = tk.StringVar(value="outputs/ui_predictions")
        self.image_size_var = tk.StringVar(value="256")
        self.device_var = tk.StringVar(value="auto")

        self.threshold_var = tk.StringVar(value="")
        self.threshold_source_var = tk.StringVar(value="Threshold source: manual / empty")

        self.hybrid_feature_weight_var = tk.StringVar(value="0.5")
        self.hybrid_ae_weight_var = tk.StringVar(value="0.5")
        self.hybrid_score_mode_var = tk.StringVar(value="max")

        self.show_contour_overlay_var = tk.BooleanVar(value=False)
        self.contour_threshold_var = tk.StringVar(value="0.50")
        self.contour_min_area_var = tk.StringVar(value="25")

        self.score_var = tk.StringVar(value="Score: -")
        self.status_var = tk.StringVar(value="Status: -")

        self.feature_engine: FeatureInferenceEngine | None = None
        self.ae_engine: AEInferenceEngine | None = None
        self.hybrid_engine: HybridInferenceEngine | None = None

        self.feature_engine_key: tuple[str, int, str] | None = None
        self.ae_engine_key: tuple[str, int, str] | None = None
        self.hybrid_engine_key: tuple[str, str, int, str, float, float, str] | None = None

        self.original_photo: ImageTk.PhotoImage | None = None
        self.overlay_photo: ImageTk.PhotoImage | None = None
        self.heatmap_photo: ImageTk.PhotoImage | None = None

        self.folder_records: list[dict] = []
        self.current_record: dict | None = None
        self.dashboard_window: DashboardWindow | None = None

        self._build_layout()
        self.try_autoload_threshold()

    def default_feature_memory_bank_path(self, category: str) -> str:
        return str(Path("outputs/checkpoints") / f"{category}_memory_bank.pt")

    def default_ae_checkpoint_path(self, category: str) -> str:
        return str(Path("outputs/ae_runs") / f"{category}_ae" / "checkpoints" / "model_best.pt")

    def threshold_json_path(self, detector_name: str, category: str) -> Path:
        return (
            Path(self.thresholds_root_var.get().strip())
            / detector_name
            / category
            / f"{category}_{detector_name}_thresholds.json"
        )

    def legacy_feature_threshold_json_path(self, category: str) -> Path:
        return Path("outputs/reports/thresholds") / category / f"{category}_thresholds.json"

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        control_frame = ttk.LabelFrame(main, text="Controls", padding=10)
        control_frame.pack(fill="x")

        ttk.Label(control_frame, text="Category").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        category_combo = ttk.Combobox(
            control_frame,
            textvariable=self.category_var,
            values=list(VALID_MVTEC_AD_CATEGORIES),
            state="readonly",
            width=14,
        )
        category_combo.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        category_combo.bind("<<ComboboxSelected>>", self.on_category_changed)

        ttk.Label(control_frame, text="Model").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        model_combo = ttk.Combobox(
            control_frame,
            textvariable=self.model_var,
            values=list(MODEL_NAMES),
            state="readonly",
            width=10,
        )
        model_combo.grid(row=0, column=3, sticky="w", padx=6, pady=6)
        model_combo.bind("<<ComboboxSelected>>", self.on_model_changed)

        ttk.Label(control_frame, text="Image Size").grid(row=0, column=4, sticky="w", padx=6, pady=6)
        ttk.Entry(control_frame, textvariable=self.image_size_var, width=8).grid(
            row=0, column=5, sticky="w", padx=6, pady=6
        )

        ttk.Label(control_frame, text="Device").grid(row=0, column=6, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            control_frame,
            textvariable=self.device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly",
            width=8,
        ).grid(row=0, column=7, sticky="w", padx=6, pady=6)

        ttk.Label(control_frame, text="Threshold").grid(row=0, column=8, sticky="w", padx=6, pady=6)
        ttk.Entry(control_frame, textvariable=self.threshold_var, width=12).grid(
            row=0, column=9, sticky="w", padx=6, pady=6
        )
        ttk.Button(control_frame, text="Auto Load", width=10, command=self.try_autoload_threshold).grid(
            row=0, column=10, sticky="ew", padx=6, pady=6
        )

        ttk.Label(control_frame, text="Feature Memory Bank").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(control_frame, textvariable=self.feature_memory_bank_var, width=84).grid(
            row=1, column=1, columnspan=9, sticky="ew", padx=6, pady=6
        )
        ttk.Button(control_frame, text="Browse", width=10, command=self.browse_feature_memory_bank).grid(
            row=1, column=10, sticky="ew", padx=6, pady=6
        )

        ttk.Label(control_frame, text="AE Checkpoint").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(control_frame, textvariable=self.ae_checkpoint_var, width=84).grid(
            row=2, column=1, columnspan=9, sticky="ew", padx=6, pady=6
        )
        ttk.Button(control_frame, text="Browse", width=10, command=self.browse_ae_checkpoint).grid(
            row=2, column=10, sticky="ew", padx=6, pady=6
        )

        ttk.Label(control_frame, text="Thresholds Root").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(control_frame, textvariable=self.thresholds_root_var, width=84).grid(
            row=3, column=1, columnspan=9, sticky="ew", padx=6, pady=6
        )
        ttk.Button(control_frame, text="Browse", width=10, command=self.browse_thresholds_root).grid(
            row=3, column=10, sticky="ew", padx=6, pady=6
        )

        ttk.Label(control_frame, text="Image").grid(row=4, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(control_frame, textvariable=self.image_path_var, width=84).grid(
            row=4, column=1, columnspan=9, sticky="ew", padx=6, pady=6
        )
        ttk.Button(control_frame, text="Browse", width=10, command=self.browse_image).grid(
            row=4, column=10, sticky="ew", padx=6, pady=6
        )

        ttk.Label(control_frame, text="Image Folder").grid(row=5, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(control_frame, textvariable=self.image_dir_var, width=84).grid(
            row=5, column=1, columnspan=9, sticky="ew", padx=6, pady=6
        )
        ttk.Button(control_frame, text="Browse", width=10, command=self.browse_image_dir).grid(
            row=5, column=10, sticky="ew", padx=6, pady=6
        )

        ttk.Label(control_frame, text="Output Dir").grid(row=6, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(control_frame, textvariable=self.output_dir_var, width=84).grid(
            row=6, column=1, columnspan=9, sticky="ew", padx=6, pady=6
        )
        ttk.Button(control_frame, text="Browse", width=10, command=self.browse_output_dir).grid(
            row=6, column=10, sticky="ew", padx=6, pady=6
        )

        ttk.Label(control_frame, text="Hybrid Feature Weight").grid(row=7, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(control_frame, textvariable=self.hybrid_feature_weight_var, width=10).grid(
            row=7, column=1, sticky="w", padx=6, pady=6
        )

        ttk.Label(control_frame, text="Hybrid AE Weight").grid(row=7, column=2, sticky="w", padx=6, pady=6)
        ttk.Entry(control_frame, textvariable=self.hybrid_ae_weight_var, width=10).grid(
            row=7, column=3, sticky="w", padx=6, pady=6
        )

        ttk.Label(control_frame, text="Hybrid Score Mode").grid(row=7, column=4, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            control_frame,
            textvariable=self.hybrid_score_mode_var,
            values=["max", "mean"],
            state="readonly",
            width=10,
        ).grid(row=7, column=5, sticky="w", padx=6, pady=6)

        ttk.Checkbutton(
            control_frame,
            text="Show contour overlay",
            variable=self.show_contour_overlay_var,
            command=self.refresh_current_record,
        ).grid(row=7, column=6, columnspan=2, sticky="w", padx=6, pady=6)

        ttk.Label(control_frame, text="Contour Threshold").grid(row=7, column=8, sticky="w", padx=6, pady=6)
        ttk.Entry(control_frame, textvariable=self.contour_threshold_var, width=10).grid(
            row=7, column=9, sticky="w", padx=6, pady=6
        )

        ttk.Label(control_frame, text="Min Area").grid(row=7, column=10, sticky="w", padx=6, pady=6)
        ttk.Entry(control_frame, textvariable=self.contour_min_area_var, width=8).grid(
            row=7, column=11, sticky="w", padx=6, pady=6
        )

        ttk.Button(control_frame, text="Run Image", command=self.run_single_inference).grid(
            row=8, column=0, columnspan=2, sticky="ew", padx=6, pady=10
        )
        ttk.Button(control_frame, text="Run Folder", command=self.run_folder_inference).grid(
            row=8, column=2, columnspan=2, sticky="ew", padx=6, pady=10
        )
        ttk.Button(control_frame, text="Open Dashboards", command=self.open_dashboard_window).grid(
            row=8, column=4, columnspan=2, sticky="ew", padx=6, pady=10
        )

        ttk.Label(
            control_frame,
            textvariable=self.score_var,
            font=(GUI_FONT_FAMILY, GUI_STATUS_FONT_SIZE, "bold"),
        ).grid(row=8, column=6, columnspan=2, sticky="w", padx=6, pady=10)

        ttk.Label(
            control_frame,
            textvariable=self.status_var,
            font=(GUI_FONT_FAMILY, GUI_STATUS_FONT_SIZE, "bold"),
        ).grid(row=8, column=8, columnspan=2, sticky="w", padx=6, pady=10)

        ttk.Label(control_frame, textvariable=self.threshold_source_var).grid(
            row=8, column=10, columnspan=2, sticky="w", padx=6, pady=10
        )

        for col in range(12):
            control_frame.columnconfigure(col, weight=0)
        control_frame.columnconfigure(1, weight=1)

        viewer_frame = ttk.Frame(main)
        viewer_frame.pack(fill="both", expand=True, pady=(12, 0))

        self.original_panel = self._create_image_panel(viewer_frame, "Original")
        self.overlay_panel = self._create_image_panel(viewer_frame, "Overlay")
        self.heatmap_panel = self._create_image_panel(viewer_frame, "Anomaly Map")

        self.original_panel.grid(row=0, column=0, sticky="nsew", padx=6)
        self.overlay_panel.grid(row=0, column=1, sticky="nsew", padx=6)
        self.heatmap_panel.grid(row=0, column=2, sticky="nsew", padx=6)

        viewer_frame.columnconfigure(0, weight=1)
        viewer_frame.columnconfigure(1, weight=1)
        viewer_frame.columnconfigure(2, weight=1)
        viewer_frame.rowconfigure(0, weight=1)

        folder_frame = ttk.LabelFrame(main, text="Folder Results", padding=10)
        folder_frame.pack(fill="both", expand=True, pady=(12, 0))

        self.results_tree = ttk.Treeview(
            folder_frame,
            columns=("rank", "name", "score", "status", "model"),
            show="headings",
            height=10,
        )
        self.results_tree.heading("rank", text="Rank")
        self.results_tree.heading("name", text="Image")
        self.results_tree.heading("score", text="Score")
        self.results_tree.heading("status", text="Status")
        self.results_tree.heading("model", text="Model")
        self.results_tree.column("rank", width=70, anchor="center")
        self.results_tree.column("name", width=500, anchor="w")
        self.results_tree.column("score", width=130, anchor="center")
        self.results_tree.column("status", width=130, anchor="center")
        self.results_tree.column("model", width=100, anchor="center")
        self.results_tree.pack(side="left", fill="both", expand=True)
        self.results_tree.bind("<<TreeviewSelect>>", self.on_folder_result_selected)

        tree_scroll = ttk.Scrollbar(folder_frame, orient="vertical", command=self.results_tree.yview)
        tree_scroll.pack(side="right", fill="y")
        self.results_tree.configure(yscrollcommand=tree_scroll.set)

        result_frame = ttk.LabelFrame(main, text="Result Summary", padding=10)
        result_frame.pack(fill="both", expand=False, pady=(12, 0))

        self.result_text = tk.Text(
            result_frame,
            height=12,
            wrap="word",
            font=(GUI_FIXED_FONT_FAMILY, GUI_FIXED_FONT_SIZE),
        )
        self.result_text.pack(fill="both", expand=True)
        self.result_text.insert("1.0", "Ready.\n")
        self.result_text.configure(state="disabled")

        self.update_model_ui_state()

    def _create_image_panel(self, parent: ttk.Frame, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        label = ttk.Label(frame, text="No image", anchor="center")
        label.pack(fill="both", expand=True)
        frame.image_label = label
        return frame

    def on_category_changed(self, _event=None) -> None:
        category = self.category_var.get().strip()
        self.feature_memory_bank_var.set(self.default_feature_memory_bank_path(category))
        self.ae_checkpoint_var.set(self.default_ae_checkpoint_path(category))
        self.feature_engine = None
        self.ae_engine = None
        self.hybrid_engine = None
        self.feature_engine_key = None
        self.ae_engine_key = None
        self.hybrid_engine_key = None
        self.try_autoload_threshold()
        if self.dashboard_window is not None and self.dashboard_window.window.winfo_exists():
            self.dashboard_window.sync_selection(category, self.current_model_name())

    def on_model_changed(self, _event=None) -> None:
        self.try_autoload_threshold()
        if self.dashboard_window is not None and self.dashboard_window.window.winfo_exists():
            self.dashboard_window.sync_selection(self.category_var.get().strip(), self.current_model_name())

    def update_model_ui_state(self) -> None:
        model_name = self.model_var.get().strip().lower()
        self.threshold_source_var.set(f"Threshold source: {model_name} manual / empty")

    def browse_feature_memory_bank(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Feature Memory Bank",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.feature_memory_bank_var.set(path)
            self.feature_engine = None
            self.hybrid_engine = None
            self.feature_engine_key = None
            self.hybrid_engine_key = None

    def browse_ae_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Select AE Checkpoint",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.ae_checkpoint_var.set(path)
            self.ae_engine = None
            self.hybrid_engine = None
            self.ae_engine_key = None
            self.hybrid_engine_key = None

    def browse_thresholds_root(self) -> None:
        path = filedialog.askdirectory(title="Select Thresholds Root")
        if path:
            self.thresholds_root_var.set(path)
            self.try_autoload_threshold()

    def browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.image_path_var.set(path)

    def browse_image_dir(self) -> None:
        path = filedialog.askdirectory(title="Select Image Folder")
        if path:
            self.image_dir_var.set(path)

    def browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir_var.set(path)

    def load_threshold_json(self, detector_name: str, category: str) -> dict | None:
        json_path = self.threshold_json_path(detector_name, category)
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))

        if detector_name == "feature":
            legacy_path = self.legacy_feature_threshold_json_path(category)
            if legacy_path.exists():
                return json.loads(legacy_path.read_text(encoding="utf-8"))

        return None

    def try_autoload_threshold(self) -> None:
        model_name = self.model_var.get().strip().lower()
        category = self.category_var.get().strip()

        payload = self.load_threshold_json(model_name, category)
        if payload is None:
            self.threshold_source_var.set(f"Threshold source: {model_name} threshold not found")
            return

        recommended = payload.get("recommended_for_gui", {})
        threshold = recommended.get("threshold")
        method = recommended.get("method", "unknown")

        if threshold is None:
            self.threshold_source_var.set(f"Threshold source: invalid {model_name} threshold json")
            return

        self.threshold_var.set(f"{float(threshold):.6f}")
        self.threshold_source_var.set(f"Threshold source: {model_name} {method}")

    def parse_threshold(self) -> float | None:
        value = self.threshold_var.get().strip()
        if not value:
            return None
        return float(value)

    def parse_contour_threshold(self) -> float:
        return float(self.contour_threshold_var.get().strip())

    def parse_contour_min_area(self) -> float:
        return float(self.contour_min_area_var.get().strip())

    def hybrid_feature_weight(self) -> float:
        return float(self.hybrid_feature_weight_var.get().strip())

    def hybrid_ae_weight(self) -> float:
        return float(self.hybrid_ae_weight_var.get().strip())

    def current_model_name(self) -> str:
        return self.model_var.get().strip().lower()

    def get_feature_engine(self) -> FeatureInferenceEngine:
        memory_bank = self.feature_memory_bank_var.get().strip()
        image_size = int(self.image_size_var.get().strip())
        device = self.device_var.get().strip()

        key = (memory_bank, image_size, device)
        if self.feature_engine is None or self.feature_engine_key != key:
            self.feature_engine = FeatureInferenceEngine(
                memory_bank_path=memory_bank,
                image_size=image_size,
                device=device,
            )
            self.feature_engine_key = key
        return self.feature_engine

    def get_ae_engine(self) -> AEInferenceEngine:
        checkpoint = self.ae_checkpoint_var.get().strip()
        image_size = int(self.image_size_var.get().strip())
        device = self.device_var.get().strip()

        key = (checkpoint, image_size, device)
        if self.ae_engine is None or self.ae_engine_key != key:
            self.ae_engine = AEInferenceEngine(
                checkpoint_path=checkpoint,
                image_size=image_size,
                device=device,
            )
            self.ae_engine_key = key
        return self.ae_engine

    def get_hybrid_engine(self) -> HybridInferenceEngine:
        feature_memory_bank = self.feature_memory_bank_var.get().strip()
        ae_checkpoint = self.ae_checkpoint_var.get().strip()
        image_size = int(self.image_size_var.get().strip())
        device = self.device_var.get().strip()
        feature_weight = self.hybrid_feature_weight()
        ae_weight = self.hybrid_ae_weight()
        score_mode = self.hybrid_score_mode_var.get().strip()

        key = (
            feature_memory_bank,
            ae_checkpoint,
            image_size,
            device,
            feature_weight,
            ae_weight,
            score_mode,
        )
        if self.hybrid_engine is None or self.hybrid_engine_key != key:
            self.hybrid_engine = HybridInferenceEngine(
                feature_memory_bank_path=feature_memory_bank,
                ae_checkpoint_path=ae_checkpoint,
                image_size=image_size,
                device=device,
                feature_weight=feature_weight,
                ae_weight=ae_weight,
                score_mode=score_mode,
            )
            self.hybrid_engine_key = key
        return self.hybrid_engine

    def set_panel_image(self, panel: ttk.LabelFrame, image: Image.Image, slot_name: str) -> None:
        image = resize_for_display(image, DISPLAY_IMAGE_SIZE)
        photo = ImageTk.PhotoImage(image)
        panel.image_label.configure(image=photo, text="")
        panel.image_label.image = photo
        setattr(self, slot_name, photo)

    def append_result_text(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def clear_folder_results(self) -> None:
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.folder_records = []
        self.current_record = None

    def status_from_score(self, score: float, threshold: float | None) -> str:
        if threshold is None:
            return "SCORED_ONLY"
        return "ANOMALY" if score >= threshold else "NORMAL"

    def require_paths_for_model(self, model_name: str) -> None:
        if model_name == "feature":
            if not Path(self.feature_memory_bank_var.get().strip()).exists():
                raise FileNotFoundError(f"Feature memory bank not found: {self.feature_memory_bank_var.get().strip()}")
        elif model_name == "ae":
            if not Path(self.ae_checkpoint_var.get().strip()).exists():
                raise FileNotFoundError(f"AE checkpoint not found: {self.ae_checkpoint_var.get().strip()}")
        elif model_name == "hybrid":
            if not Path(self.feature_memory_bank_var.get().strip()).exists():
                raise FileNotFoundError(f"Feature memory bank not found: {self.feature_memory_bank_var.get().strip()}")
            if not Path(self.ae_checkpoint_var.get().strip()).exists():
                raise FileNotFoundError(f"AE checkpoint not found: {self.ae_checkpoint_var.get().strip()}")

    def build_record_from_single_result(self, model_name: str, result, exports: dict[str, str], threshold: float | None) -> dict:
        record = {
            "model_name": model_name,
            "image_path": result.image_path,
            "category": result.category,
            "image_score": float(result.image_score),
            "threshold": threshold,
            "status": self.status_from_score(float(result.image_score), threshold),
            "image_size_hw": list(result.image_size_hw),
            "anomaly_map": None,
            "exports": exports,
            "extra": {},
        }

        if model_name == "feature":
            record["anomaly_map"] = result.anomaly_map
            record["extra"] = {
                "patch_grid_size_hw": list(result.patch_grid_size_hw),
            }
        elif model_name == "ae":
            record["anomaly_map"] = result.anomaly_map
        else:
            record["anomaly_map"] = result.fused_anomaly_map
            record["extra"] = {
                "feature_score": float(result.feature_score),
                "ae_score": float(result.ae_score),
                "fusion_weight_feature": float(result.fusion_weight_feature),
                "fusion_weight_ae": float(result.fusion_weight_ae),
            }

        return record

    def run_single_engine_predict(self, model_name: str, image_path: str, output_dir: Path) -> dict:
        threshold = self.parse_threshold()

        if model_name == "feature":
            engine = self.get_feature_engine()
            result = engine.predict_image(image_path)
            exports = export_inference_result(result=result, output_dir=output_dir)
            return self.build_record_from_single_result(model_name, result, exports, threshold)

        if model_name == "ae":
            engine = self.get_ae_engine()
            result = engine.predict_image(image_path)
            exports = export_ae_inference_result(result=result, output_dir=output_dir)
            return self.build_record_from_single_result(model_name, result, exports, threshold)

        engine = self.get_hybrid_engine()
        result = engine.predict_image(image_path)
        exports = export_hybrid_inference_result(result=result, output_dir=output_dir)
        return self.build_record_from_single_result(model_name, result, exports, threshold)

    def build_overlay_image_for_record(self, record: dict) -> Image.Image:
        overlay_path = record["exports"].get("overlay_png") or record["exports"].get("hybrid_overlay_png")
        if overlay_path is None:
            raise ValueError("Overlay image export not found.")

        overlay_rgb = np.asarray(Image.open(overlay_path).convert("RGB"), dtype=np.uint8)

        if not self.show_contour_overlay_var.get():
            return Image.fromarray(overlay_rgb, mode="RGB")

        contours, _, _ = extract_anomaly_contours(
            anomaly_map=record["anomaly_map"],
            threshold=self.parse_contour_threshold(),
            min_area=self.parse_contour_min_area(),
            blur_kernel=0,
            morph_kernel=3,
            morph_iterations=1,
        )
        contour_overlay_rgb = draw_anomaly_contours(
            image_rgb=overlay_rgb,
            contours=contours,
            line_thickness=2,
            draw_fill=False,
        )
        return Image.fromarray(contour_overlay_rgb, mode="RGB")

    def show_result_record(self, record: dict) -> None:
        self.current_record = record

        original_image = load_rgb_image(record["image_path"])
        overlay_image = self.build_overlay_image_for_record(record)
        heat_image = anomaly_map_to_heat_image(record["anomaly_map"])

        self.set_panel_image(self.original_panel, original_image, "original_photo")
        self.set_panel_image(self.overlay_panel, overlay_image, "overlay_photo")
        self.set_panel_image(self.heatmap_panel, heat_image, "heatmap_photo")

        self.score_var.set(f"Score: {record['image_score']:.6f}")

        threshold = record["threshold"]
        if threshold is None:
            self.status_var.set(f"Status: {record['status']}")
        else:
            self.status_var.set(f"Status: {record['status']}  (threshold={threshold:.6f})")

        summary = {
            "model_name": record["model_name"],
            "image_path": record["image_path"],
            "category": record["category"],
            "image_score": record["image_score"],
            "threshold": record["threshold"],
            "status": record["status"],
            "show_contour_overlay": bool(self.show_contour_overlay_var.get()),
            "contour_threshold": self.parse_contour_threshold(),
            "contour_min_area": self.parse_contour_min_area(),
            "image_size_hw": record["image_size_hw"],
            "extra": record["extra"],
            "exports": record["exports"],
        }
        self.append_result_text(json.dumps(summary, indent=2))

    def refresh_current_record(self) -> None:
        if self.current_record is not None:
            try:
                self.show_result_record(self.current_record)
            except Exception as exc:
                messagebox.showerror("Contour Overlay Error", str(exc))

    def open_dashboard_window(self) -> None:
        if self.dashboard_window is None or not self.dashboard_window.window.winfo_exists():
            self.dashboard_window = DashboardWindow(
                self.root,
                category=self.category_var.get().strip(),
                detector=self.current_model_name(),
            )
        else:
            self.dashboard_window.window.lift()
            self.dashboard_window.sync_selection(
                self.category_var.get().strip(),
                self.current_model_name(),
            )

    def run_single_inference(self) -> None:
        try:
            image_path = self.image_path_var.get().strip()
            model_name = self.current_model_name()
            category = self.category_var.get().strip()
            output_dir = Path(self.output_dir_var.get().strip()) / category / model_name / "single_image"

            if not image_path:
                raise ValueError("Please select an image.")
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            self.require_paths_for_model(model_name)
            self.clear_folder_results()

            record = self.run_single_engine_predict(
                model_name=model_name,
                image_path=image_path,
                output_dir=output_dir,
            )
            self.show_result_record(record)

        except Exception as exc:
            messagebox.showerror("Inference Error", str(exc))

    def run_folder_inference(self) -> None:
        try:
            image_dir = self.image_dir_var.get().strip()
            model_name = self.current_model_name()
            category = self.category_var.get().strip()

            if not image_dir:
                raise ValueError("Please select an image folder.")
            if not Path(image_dir).exists():
                raise FileNotFoundError(f"Image folder not found: {image_dir}")

            self.require_paths_for_model(model_name)
            self.clear_folder_results()

            folder_name = Path(image_dir).name
            output_dir = Path(self.output_dir_var.get().strip()) / category / model_name / folder_name

            image_paths = sorted(
                p
                for p in Path(image_dir).iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
            )

            records: list[dict] = []
            for image_path in image_paths:
                records.append(
                    self.run_single_engine_predict(
                        model_name=model_name,
                        image_path=str(image_path),
                        output_dir=output_dir,
                    )
                )

            records.sort(key=lambda x: x["image_score"], reverse=True)
            self.folder_records = records

            for idx, record in enumerate(records, start=1):
                self.results_tree.insert(
                    "",
                    "end",
                    iid=str(idx - 1),
                    values=(
                        idx,
                        Path(record["image_path"]).name,
                        f"{record['image_score']:.6f}",
                        record["status"],
                        record["model_name"],
                    ),
                )

            if records:
                self.results_tree.selection_set("0")
                self.results_tree.focus("0")
                self.show_result_record(records[0])

        except Exception as exc:
            messagebox.showerror("Folder Inference Error", str(exc))

    def on_folder_result_selected(self, _event=None) -> None:
        selected = self.results_tree.selection()
        if not selected:
            return
        idx = int(selected[0])
        if 0 <= idx < len(self.folder_records):
            self.show_result_record(self.folder_records[idx])


def main() -> None:
    root = tk.Tk()
    app = AnomalyTkinterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()