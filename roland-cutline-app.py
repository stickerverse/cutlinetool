import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QSlider, QGroupBox,
                           QToolBar, QAction, QFileDialog, QSplitter, QComboBox,
                           QSpinBox, QCheckBox, QTabWidget, QListWidget, QDockWidget,
                           QGraphicsScene, QGraphicsView, QGraphicsPixmapItem,
                           QStatusBar, QMenuBar, QMenu, QToolButton, QButtonGroup,
                           QRadioButton, QDoubleSpinBox, QFrame, QProgressDialog,
                           QListWidgetItem, QLineEdit, QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QRectF, QTimer, QSize, QThread, QSettings
from PyQt5.QtGui import QPixmap, QPainter, QColor, QBrush, QPen, QIcon, QImage, QPainterPath
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import pyclipper
import traceback
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import requests
import hashlib

# Model configurations
ESRGAN_MODELS = {
    'RealESRGAN_x4plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'scale': 4,
        'model_arch': 'RRDBNet',
        'num_block': 23,
        'description': 'General purpose 4x upscaling'
    },
    'RealESRGAN_x4plus_anime_6B': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
        'scale': 4,
        'model_arch': 'RRDBNet',
        'num_block': 6,
        'description': 'Optimized for anime/illustrations (smaller model)'
    },
    'RealESRGAN_x2plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        'scale': 2,
        'model_arch': 'RRDBNet',
        'num_block': 23,
        'description': 'General purpose 2x upscaling'
    }
}

# Model download/management
class ModelManager:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
    def get_model_path(self, model_name):
        return os.path.join(self.models_dir, f"{model_name}.pth")
    
    def is_model_downloaded(self, model_name):
        return os.path.exists(self.get_model_path(model_name))
    
    def download_model(self, model_name, progress_callback=None):
        if model_name not in ESRGAN_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_info = ESRGAN_MODELS[model_name]
        url = model_info['url']
        output_path = self.get_model_path(model_name)
        
        if self.is_model_downloaded(model_name):
            return output_path
            
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    progress = int((downloaded / total_size) * 100)
                    progress_callback(progress, f"Downloading {model_name}... {progress}%")
                    
        return output_path

# Thread for Real-ESRGAN upscaling
class UpscalingThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(np.ndarray, float)  # result array, actual scale
    error = pyqtSignal(str)
    
    def __init__(self, image_array, model_name, target_scale, tile_size=0):
        super().__init__()
        self.image_array = image_array
        self.model_name = model_name
        self.target_scale = target_scale
        self.tile_size = tile_size
        self.model_manager = ModelManager()
        
    def run(self):
        try:
            self.progress.emit(5, "Checking model...")
            
            # Download model if needed
            if not self.model_manager.is_model_downloaded(self.model_name):
                self.progress.emit(10, f"Downloading {self.model_name}...")
                model_path = self.model_manager.download_model(
                    self.model_name, 
                    lambda p, t: self.progress.emit(int(p * 0.3), t)
                )
            else:
                model_path = self.model_manager.get_model_path(self.model_name)
                
            self.progress.emit(35, "Loading model...")
            
            # Get model configuration
            model_info = ESRGAN_MODELS[self.model_name]
            
            # Initialize model
            if model_info['model_arch'] == 'RRDBNet':
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                               num_block=model_info['num_block'], num_grow_ch=32, 
                               scale=model_info['scale'])
            
            self.progress.emit(45, "Initializing upscaler...")
            
            # Determine device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize upscaler
            upsampler = RealESRGANer(
                scale=model_info['scale'],
                model_path=model_path,
                model=model,
                tile=512,  # Use tiling with 512x512 tiles to avoid memory issues
                tile_pad=10,
                pre_pad=0,
                half=True if device.type == 'cuda' else False,
                device=device
            )
            
            self.progress.emit(60, "Processing image...")
            
            # Convert image to format expected by Real-ESRGAN
            if len(self.image_array.shape) == 2:
                # Grayscale to RGB
                img = cv2.cvtColor(self.image_array, cv2.COLOR_GRAY2RGB)
            elif self.image_array.shape[2] == 4:
                # RGBA to RGB
                img = cv2.cvtColor(self.image_array, cv2.COLOR_RGBA2RGB)
            else:
                img = self.image_array
                
            # Upscale
            output, _ = upsampler.enhance(img, outscale=self.target_scale/model_info['scale'])
            
            self.progress.emit(90, "Finalizing...")
            
            # Calculate actual scale achieved
            actual_scale = output.shape[0] / self.image_array.shape[0]
            
            self.progress.emit(100, "Complete!")
            self.finished.emit(output, actual_scale)
            
        except Exception as e:
            self.error.emit(f"Upscaling failed: {str(e)}\n{traceback.format_exc()}")

# Comparison dialog for before/after
class ComparisonDialog(QDialog):
    def __init__(self, original_pixmap, upscaled_pixmap, actual_scale, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Upscaling Comparison")
        self.setModal(True)
        self.resize(1200, 700)
        
        self.original_pixmap = original_pixmap
        self.upscaled_pixmap = upscaled_pixmap
        self.actual_scale = actual_scale
        self.split_position = 0.5
        self.use_upscaled = False
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel(f"Original: {self.original_pixmap.width()}x{self.original_pixmap.height()} → "
                           f"Upscaled: {self.upscaled_pixmap.width()}x{self.upscaled_pixmap.height()} "
                           f"(Scale: {self.actual_scale:.2f}x)")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 16px; padding: 10px;")
        layout.addWidget(info_label)
        
        # Comparison view
        self.comparison_view = QGraphicsView()
        self.comparison_scene = QGraphicsScene()
        self.comparison_view.setScene(self.comparison_scene)
        self.comparison_view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.comparison_view, stretch=1)
        
        # Slider for split position
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Original"))
        
        self.split_slider = QSlider(Qt.Horizontal)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.valueChanged.connect(self.update_split)
        slider_layout.addWidget(self.split_slider, stretch=1)
        
        slider_layout.addWidget(QLabel("Upscaled"))
        layout.addLayout(slider_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.use_original_btn = QPushButton("Use Original")
        self.use_original_btn.clicked.connect(self.use_original)
        button_layout.addWidget(self.use_original_btn)
        
        self.use_upscaled_btn = QPushButton("Use Upscaled")
        self.use_upscaled_btn.setStyleSheet("QPushButton { background-color: #667eea; color: white; font-weight: bold; }")
        self.use_upscaled_btn.clicked.connect(self.use_upscaled_result)
        button_layout.addWidget(self.use_upscaled_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Initial draw
        self.update_comparison()
        
    def update_split(self, value):
        self.split_position = value / 100.0
        self.update_comparison()
        
    def update_comparison(self):
        self.comparison_scene.clear()
        
        # Scale pixmaps to fit view
        view_size = self.comparison_view.size()
        scaled_original = self.original_pixmap.scaled(
            view_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled_upscaled = self.upscaled_pixmap.scaled(
            view_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Create split view
        width = scaled_original.width()
        height = scaled_original.height()
        split_x = int(width * self.split_position)
        
        # Combined pixmap
        combined = QPixmap(width, height)
        painter = QPainter(combined)
        
        # Draw original on left
        painter.drawPixmap(0, 0, split_x, height, 
                          scaled_original, 0, 0, split_x, height)
        
        # Draw upscaled on right
        painter.drawPixmap(split_x, 0, width - split_x, height,
                          scaled_upscaled, split_x, 0, width - split_x, height)
        
        # Draw split line
        painter.setPen(QPen(Qt.white, 2))
        painter.drawLine(split_x, 0, split_x, height)
        
        painter.end()
        
        # Add to scene
        self.comparison_scene.addPixmap(combined)
        self.comparison_view.fitInView(self.comparison_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_comparison()
        
    def use_original(self):
        self.use_upscaled = False
        self.accept()
        
    def use_upscaled_result(self):
        self.use_upscaled = True
        self.accept()

# Thread for background removal
class BackgroundRemovalThread(QThread):
    progress = pyqtSignal(int, str)  # progress value, status text
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self, image_array, use_alpha_matting=True):
        super().__init__()
        self.image_array = image_array
        self.use_alpha_matting = use_alpha_matting
        
    def run(self):
        try:
            self.progress.emit(10, "Converting image format...")
            pil_img = Image.fromarray(self.image_array, 'RGBA')
            
            self.progress.emit(30, "Initializing AI model...")
            
            # Remove background with optional alpha matting
            if self.use_alpha_matting:
                self.progress.emit(50, "Removing background with alpha matting...")
                out_img = remove(pil_img, alpha_matting=True, 
                               alpha_matting_foreground_threshold=240,
                               alpha_matting_background_threshold=10,
                               alpha_matting_erode_size=10)
            else:
                self.progress.emit(50, "Removing background...")
                out_img = remove(pil_img)
            
            self.progress.emit(90, "Finalizing...")
            result_array = np.array(out_img)
            
            self.progress.emit(100, "Complete!")
            self.finished.emit(result_array)
            
        except Exception as e:
            self.error.emit(f"Background removal failed: {str(e)}")

# Thread for cut line generation
class CutLineGenerationThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict, dict)  # cutline_dict, color_dict
    error = pyqtSignal(str)
    
    def __init__(self, image_array, offset_px, smooth_val, selected_type):
        super().__init__()
        self.image_array = image_array
        self.offset_px = offset_px
        self.smooth_val = smooth_val
        self.selected_type = selected_type
        
    def run(self):
        try:
            self.progress.emit(10, "Converting to grayscale...")
            gray = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
            
            self.progress.emit(20, "Detecting edges...")
            edges = cv2.Canny(gray, 50, 150)
            
            self.progress.emit(30, "Applying offset...")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.offset_px*2, self.offset_px*2))
            offset_edges = cv2.dilate(edges, kernel, iterations=1)
            
            self.progress.emit(40, "Finding contours...")
            contours, _ = cv2.findContours(offset_edges, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            self.progress.emit(50, "Processing contours...")
            scale = 1000
            pc = pyclipper.PyclipperOffset()
            cutline_dict = {'CutContour': [], 'PerfCutContour': []}
            
            total_contours = len(contours)
            for idx, cnt in enumerate(contours):
                if len(cnt) < 3:
                    continue
                    
                progress_val = 50 + int((idx / total_contours) * 40)
                self.progress.emit(progress_val, f"Processing contour {idx+1}/{total_contours}")
                
                epsilon = self.smooth_val
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                path = [(int(pt[0][0]*scale), int(pt[0][1]*scale)) for pt in approx]
                
                pc.Clear()
                
                if self.selected_type == 'CutContour':
                    pc.AddPath(path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    offset_paths = pc.Execute(self.offset_px*scale)
                    
                    for opath in offset_paths:
                        opath_f = [(pt[0]/scale, pt[1]/scale) for pt in opath]
                        qpath = QPainterPath()
                        if opath_f:
                            qpath.moveTo(opath_f[0][0], opath_f[0][1])
                            for pt in opath_f[1:]:
                                qpath.lineTo(pt[0], pt[1])
                            qpath.closeSubpath()
                            cutline_dict['CutContour'].append(qpath)
                            
                elif self.selected_type == 'PerfCutContour':
                    perf_offset_px = int(self.offset_px * 0.7)
                    if perf_offset_px < 1:
                        perf_offset_px = 1
                    pc.AddPath(path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    perf_paths = pc.Execute(perf_offset_px*scale)
                    
                    for opath in perf_paths:
                        opath_f = [(pt[0]/scale, pt[1]/scale) for pt in opath]
                        qpath = QPainterPath()
                        if opath_f:
                            qpath.moveTo(opath_f[0][0], opath_f[0][1])
                            for pt in opath_f[1:]:
                                qpath.lineTo(pt[0], pt[1])
                            qpath.closeSubpath()
                            cutline_dict['PerfCutContour'].append(qpath)
            
            self.progress.emit(95, "Finalizing cut lines...")
            
            color_dict = {
                'CutContour': QColor(236, 0, 140),
                'PerfCutContour': QColor(60, 180, 75)
            }
            
            self.progress.emit(100, "Complete!")
            self.finished.emit(cutline_dict, color_dict)
            
        except Exception as e:
            self.error.emit(f"Cut line generation failed: {str(e)}\n{traceback.format_exc()}")

class LayerListItem(QWidget):
    def __init__(self, name, pixmap, locked=False):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(8)
        self.thumb = QLabel()
        self.thumb.setPixmap(pixmap.scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.thumb.setFixedSize(36, 36)
        layout.addWidget(self.thumb)
        self.label = QLabel(name)
        self.label.setStyleSheet("color: #e0e0e0; font-size: 15px; font-weight: 500;")
        layout.addWidget(self.label)
        if locked:
            lock_icon = QIcon.fromTheme("object-locked")
            lock_label = QLabel()
            lock_label.setPixmap(lock_icon.pixmap(16, 16))
            layout.addWidget(lock_label)
        layout.addStretch(1)
        self.setLayout(layout)
        self.setStyleSheet("background: #2d313a; border-radius: 6px; border: 1px solid #444;")
        self.setMinimumHeight(40)
        self.setMaximumHeight(44)

class LayersPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.label = QLabel("Layers")
        self.label.setStyleSheet("font-weight: bold; margin-bottom: 4px; color: #e0e0e0; font-size: 16px;")
        layout.addWidget(self.label)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SingleSelection)
        self.list.setStyleSheet("QListWidget { background: #23272e; border: 1.5px solid #3a3f4b; border-radius: 8px; color: #e0e0e0; font-size: 15px; }"
                               "QListWidget::item:selected { background: #3a3f4b; border: 2px solid #667eea; color: #fff; }")
        layout.addWidget(self.list, stretch=1)
        # Layer controls (add, delete, up, down)
        controls_layout = QHBoxLayout()
        self.add_btn = QPushButton(QIcon("/usr/share/icons/breeze/actions/24/list-add.svg"), "")
        self.add_btn.setToolTip("Add new layer")
        self.del_btn = QPushButton(QIcon("/usr/share/icons/breeze/actions/24/list-remove.svg"), "")
        self.del_btn.setToolTip("Delete selected layer")
        self.up_btn = QPushButton(QIcon("/usr/share/icons/breeze/actions/24/go-up.svg"), "")
        self.up_btn.setToolTip("Move layer up")
        self.down_btn = QPushButton(QIcon("/usr/share/icons/breeze/actions/24/go-down.svg"), "")
        self.down_btn.setToolTip("Move layer down")
        for btn in [self.add_btn, self.del_btn, self.up_btn, self.down_btn]:
            btn.setFixedSize(28, 28)
            btn.setStyleSheet("QPushButton { background: #ffffff; border: 1px solid #444; border-radius: 6px; } QPushButton:hover { background: #e0e0e0; }")
        controls_layout.addWidget(self.add_btn)
        controls_layout.addWidget(self.del_btn)
        controls_layout.addWidget(self.up_btn)
        controls_layout.addWidget(self.down_btn)
        layout.addLayout(controls_layout)
        self.setLayout(layout)
        self.setMinimumHeight(350)
        self.setMaximumHeight(600)
        self.setStyleSheet("background: #23272e; border-top: 2px solid #3a3f4b;")
        self.add_btn.clicked.connect(self.add_new_layer)
        self.del_btn.clicked.connect(self.delete_selected_layer)
        self.up_btn.clicked.connect(self.move_layer_up)
        self.down_btn.clicked.connect(self.move_layer_down)
        self.list.itemDoubleClicked.connect(self.rename_layer)
        self.background_icon = QIcon.fromTheme("object-locked")
        self._init_background_layer()

    def _init_background_layer(self):
        self.clear_layers()
        # White thumbnail for background
        bg_pixmap = QPixmap(32, 32)
        bg_pixmap.fill(Qt.white)
        item = QListWidgetItem()
        widget = LayerListItem("Background", bg_pixmap, locked=True)
        item.setSizeHint(widget.sizeHint())
        self.list.addItem(item)
        self.list.setItemWidget(item, widget)
        self.list.setCurrentItem(item)

    def add_image_layer(self, name, image):
        pixmap = QPixmap.fromImage(image).scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        item = QListWidgetItem()
        widget = LayerListItem(name, pixmap)
        item.setSizeHint(widget.sizeHint())
        self.list.insertItem(0, item)
        self.list.setItemWidget(item, widget)
        self.list.setCurrentItem(item)

    def add_cutline_layer(self, name, qpaths, scene_rect):
        # Render cutline to pixmap
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        scale = min(32/scene_rect.width(), 32/scene_rect.height())
        painter.scale(scale, scale)
        pen = QPen(Qt.magenta if "CutContour" in name else Qt.green)
        pen.setWidth(2)
        painter.setPen(pen)
        for qpath in qpaths:
            painter.drawPath(qpath)
        painter.end()
        item = QListWidgetItem()
        widget = LayerListItem(name, pixmap)
        item.setSizeHint(widget.sizeHint())
        self.list.insertItem(0, item)
        self.list.setItemWidget(item, widget)
        self.list.setCurrentItem(item)

    def add_new_layer(self):
        name = self._unique_layer_name()
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        item = QListWidgetItem()
        widget = LayerListItem(name, pixmap)
        item.setSizeHint(widget.sizeHint())
        self.list.insertItem(0, item)
        self.list.setItemWidget(item, widget)
        self.list.setCurrentItem(item)

    def _unique_layer_name(self):
        base = "Layer"
        idx = 1
        names = [self.list.itemWidget(self.list.item(i)).label.text() for i in range(self.list.count())]
        while f"{base} {idx}" in names:
            idx += 1
        return f"{base} {idx}"

    def delete_selected_layer(self):
        item = self.list.currentItem()
        if item is None:
            return
        widget = self.list.itemWidget(item)
        if widget and widget.label.text() != "Background":
            row = self.list.row(item)
            self.list.takeItem(row)

    def move_layer_up(self):
        row = self.list.currentRow()
        if row > 0:
            item = self.list.takeItem(row)
            self.list.insertItem(row - 1, item)
            self.list.setCurrentItem(item)

    def move_layer_down(self):
        row = self.list.currentRow()
        if row < self.list.count() - 1:
            item = self.list.takeItem(row)
            self.list.insertItem(row + 1, item)
            self.list.setCurrentItem(item)

    def rename_layer(self, item):
        widget = self.list.itemWidget(item)
        if widget and widget.label.text() != "Background":
            self.list.editItem(item)

    def clear_layers(self):
        self.list.clear()

    def next_cutline_name(self, base):
        idx = 1
        names = [self.list.itemWidget(self.list.item(i)).label.text() for i in range(self.list.count())]
        while f"{base} {idx}" in names:
            idx += 1
        return f"{base} {idx}"

class Canvas(QGraphicsView):
    """Main canvas for displaying and editing images with cut lines"""
    
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Canvas settings
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setMouseTracking(True)
        
        # Canvas state
        self.image_item = None
        self.cut_lines = {'CutContour': [], 'PerfCutContour': []}
        self.current_zoom = 1.0
        self.show_cut_lines = {'CutContour': True, 'PerfCutContour': True}
        self.show_original = True
        
        # Background pattern
        self.setBackgroundBrush(QBrush(Qt.gray, Qt.Dense6Pattern))
        
    def load_image(self, image_path):
        """Load an image onto the canvas"""
        try:
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                raise Exception("Invalid image file")
                
            if self.image_item:
                self.scene.removeItem(self.image_item)
            
            self.image_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.image_item)
            self.scene.setSceneRect(self.image_item.boundingRect())
            self.fitInView(self.image_item, Qt.KeepAspectRatio)
            return True
            
        except Exception as e:
            return False
    
    def update_image(self, pixmap):
        """Update the current image with a new pixmap"""
        if self.image_item:
            self.image_item.setPixmap(pixmap)
            self.scene.setSceneRect(self.image_item.boundingRect())
            self.fitInView(self.image_item, Qt.KeepAspectRatio)
        
    def wheelEvent(self, event):
        """Handle zoom with mouse wheel"""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        
        # Save the scene pos
        old_pos = self.mapToScene(event.pos())
        
        # Zoom
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
            self.current_zoom *= zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
            self.current_zoom *= zoom_out_factor
            
        self.scale(zoom_factor, zoom_factor)
        
        # Get the new position
        new_pos = self.mapToScene(event.pos())
        
        # Move scene to old position
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def clear_cut_lines(self):
        for k in self.cut_lines:
            for item in self.cut_lines[k]:
                self.scene.removeItem(item)
            self.cut_lines[k] = []

    def draw_cut_lines(self, cutline_dict, color_dict, width=1):
        self.clear_cut_lines()
        for k, qpaths in cutline_dict.items():
            if not self.show_cut_lines.get(k, True):
                continue
            pen = QPen(color_dict[k])
            pen.setWidth(width)
            pen.setCosmetic(True)
            for qpath in qpaths:
                item = self.scene.addPath(qpath, pen)
                self.cut_lines[k].append(item)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_item:
            self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def fit_to_window(self):
        if self.image_item:
            self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def show_only_original(self, original_pixmap):
        self.scene.clear()
        self.image_item = QGraphicsPixmapItem(original_pixmap)
        self.scene.addItem(self.image_item)
        self.scene.setSceneRect(self.image_item.boundingRect())
        self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def show_only_cutlines(self, cutline_dict, color_dict, width=2):
        self.scene.clear()
        for k, qpaths in cutline_dict.items():
            pen = QPen(color_dict[k])
            pen.setWidth(width)
            pen.setCosmetic(True)
            for qpath in qpaths:
                self.scene.addPath(qpath, pen)

    def show_processed(self, pixmap, cutline_dict, color_dict, width=2):
        self.scene.clear()
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        for k, qpaths in cutline_dict.items():
            pen = QPen(color_dict[k])
            pen.setWidth(width)
            pen.setCosmetic(True)
            for qpath in qpaths:
                self.scene.addPath(qpath, pen)
        self.scene.setSceneRect(self.image_item.boundingRect())
        self.fitInView(self.image_item, Qt.KeepAspectRatio)

class ToolPanel(QWidget):
    """Left side tool panel with all processing options"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        
        # AI Enhancement Section
        ai_group = QGroupBox("AI Enhancement")
        ai_layout = QVBoxLayout()
        
        # Upscaling
        upscale_label = QLabel("Image Upscaling")
        upscale_label.setStyleSheet("font-weight: bold;")
        ai_layout.addWidget(upscale_label)
        
        self.upscale_checkbox = QCheckBox("Enable Upscaling")
        self.upscale_checkbox.setToolTip("Enable AI-powered upscaling for higher resolution.")
        ai_layout.addWidget(self.upscale_checkbox)
        
        # Model selection
        self.upscale_model = QComboBox()
        self.upscale_model.addItems([
            "RealESRGAN_x4plus (General)",
            "RealESRGAN_x4plus_anime_6B (Anime)",
            "RealESRGAN_x2plus (2x General)"
        ])
        self.upscale_model.setToolTip("Select upscaling model")
        self.upscale_model.setEnabled(False)
        ai_layout.addWidget(self.upscale_model)
        
        self.upscale_factor = QComboBox()
        self.upscale_factor.addItems(["2x", "3x", "4x"])
        self.upscale_factor.setCurrentIndex(2)  # Default to 4x
        self.upscale_factor.setToolTip("Select target upscaling factor.")
        ai_layout.addWidget(self.upscale_factor)
        
        self.upscale_btn = QPushButton("Upscale Image")
        self.upscale_btn.setToolTip("Begin the upscaling process using the selected factor.")
        ai_layout.addWidget(self.upscale_btn)
        
        self.compare_btn = QPushButton("Compare Before/After")
        self.compare_btn.setEnabled(False)
        self.compare_btn.setToolTip("Compare original and upscaled images.")
        ai_layout.addWidget(self.compare_btn)
        
        ai_layout.addSpacing(10)
        
        # Background Removal
        bg_label = QLabel("Background Removal")
        bg_label.setStyleSheet("font-weight: bold;")
        ai_layout.addWidget(bg_label)
        
        self.bg_model = QComboBox()
        self.bg_model.addItems(["U2-Net (Best Quality)", "U2-NetP (Faster)", "MODNet (Real-time)"])
        self.bg_model.setToolTip("Choose background removal model.")
        ai_layout.addWidget(self.bg_model)
        
        self.alpha_matting = QCheckBox("Enable Alpha Matting")
        self.alpha_matting.setChecked(True)
        self.alpha_matting.setToolTip("Enable advanced edge refinement for background removal.")
        ai_layout.addWidget(self.alpha_matting)
        
        self.remove_bg_btn = QPushButton("Remove Background")
        self.remove_bg_btn.setToolTip("Remove background from the loaded image.")
        ai_layout.addWidget(self.remove_bg_btn)
        
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        
        # Cut Line Settings
        cut_group = QGroupBox("Cut Line Settings")
        cut_layout = QVBoxLayout()
        
        # Offset setting (slider)
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Offset:"))
        self.offset_slider = QSlider(Qt.Horizontal)
        self.offset_slider.setMinimum(1)  # 0.01 in
        self.offset_slider.setMaximum(100)  # 1.00 in
        self.offset_slider.setValue(6)  # Default to 0.06 in
        self.offset_slider.setSingleStep(1)
        self.offset_slider.setToolTip("Set the offset for cut lines (in inches, 0.01-1.00)")
        offset_layout.addWidget(self.offset_slider)
        self.offset_label = QLabel("0.06 in")
        offset_layout.addWidget(self.offset_label)
        cut_layout.addLayout(offset_layout)
        
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Smooth Cutline:"))
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setMinimum(1)
        self.smooth_slider.setMaximum(50)
        self.smooth_slider.setValue(10)
        self.smooth_slider.setSingleStep(1)
        self.smooth_slider.setToolTip("Adjust cutline smoothness (higher = smoother)")
        smooth_layout.addWidget(self.smooth_slider)
        self.smooth_label = QLabel("10")
        smooth_layout.addWidget(self.smooth_label)
        cut_layout.addLayout(smooth_layout)
        
        self.cut_type = QComboBox()
        self.cut_type.addItems(["CutContour", "PerfCutContour"])
        self.cut_type.setToolTip("Select the type of cut line to generate.")
        cut_layout.addWidget(self.cut_type)
        
        self.generate_cut_btn = QPushButton("Generate Cut Lines")
        self.generate_cut_btn.setStyleSheet("QPushButton { background-color: #667eea; color: white; font-weight: bold; }")
        self.generate_cut_btn.setToolTip("Automatically generate cut lines around detected objects.")
        cut_layout.addWidget(self.generate_cut_btn)
        
        cut_group.setLayout(cut_layout)
        layout.addWidget(cut_group)
        
        # Object Detection
        object_group = QGroupBox("Object Detection")
        object_layout = QVBoxLayout()
        
        self.detect_objects_btn = QPushButton("Detect Multiple Objects")
        self.detect_objects_btn.setToolTip("Detect and list multiple objects in the image.")
        object_layout.addWidget(self.detect_objects_btn)
        
        self.object_list = QListWidget()
        self.object_list.setMaximumHeight(100)
        self.object_list.setToolTip("List of detected objects.")
        object_layout.addWidget(self.object_list)
        
        object_group.setLayout(object_layout)
        layout.addWidget(object_group)
        
        # Export Settings
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        self.export_pdf_btn = QPushButton("Export PDF with Cut Lines")
        self.export_pdf_btn.setStyleSheet("QPushButton { background-color: #48bb78; color: white; font-weight: bold; }")
        self.export_pdf_btn.setToolTip("Export the current image and cut lines as a PDF.")
        export_layout.addWidget(self.export_pdf_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        self.setLayout(layout)
        self.setMaximumWidth(300)
        
    def on_upscale_toggled(self, state):
        enabled = state == Qt.Checked
        self.upscale_model.setEnabled(enabled)
        self.upscale_factor.setEnabled(enabled)
        self.upscale_btn.setEnabled(enabled)

class PropertiesPanel(QWidget):
    """Right side properties panel for fine-tuning"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        
        # Image Properties
        img_group = QGroupBox("Image Properties")
        img_layout = QVBoxLayout()
        
        self.dimensions_label = QLabel("Dimensions: -")
        self.resolution_label = QLabel("Resolution: -")
        self.colorspace_label = QLabel("Color Space: -")
        
        img_layout.addWidget(self.dimensions_label)
        img_layout.addWidget(self.resolution_label)
        img_layout.addWidget(self.colorspace_label)
        
        img_group.setLayout(img_layout)
        layout.addWidget(img_group)
        
        # Processing Status
        status_group = QGroupBox("Processing Status")
        status_layout = QVBoxLayout()
        
        self.upscale_status = QLabel("Upscaling: Not applied")
        self.bg_status = QLabel("Background: Original")
        self.cutline_status = QLabel("Cut Lines: Not generated")
        
        status_layout.addWidget(self.upscale_status)
        status_layout.addWidget(self.bg_status)
        status_layout.addWidget(self.cutline_status)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Advanced Settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QVBoxLayout()
        
        # Edge refinement
        self.edge_threshold = QSlider(Qt.Horizontal)
        self.edge_threshold.setRange(0, 100)
        self.edge_threshold.setValue(50)
        self.edge_label = QLabel("Edge Sensitivity: 50")
        
        advanced_layout.addWidget(self.edge_label)
        advanced_layout.addWidget(self.edge_threshold)
        
        self.edge_threshold.valueChanged.connect(
            lambda v: self.edge_label.setText(f"Edge Sensitivity: {v}")
        )
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        self.setLayout(layout)
        self.setMaximumWidth(250)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Roland Cut Line Tool - Professional Edition")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize settings
        self.settings = QSettings("YourCompany", "RolandCutLineTool")
        
        # Initialize thread references
        self.bg_removal_thread = None
        self.cutline_thread = None
        self.upscaling_thread = None
        self.current_progress_dialog = None
        
        # Initialize upscaling state
        self.last_upscaled_result = None
        self.upscale_applied = False

        # Apply a modern dark theme stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #23272e;
            }
            QWidget {
                color: #e0e0e0;
                background-color: #23272e;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 14px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1.5px solid #3a3f4b;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #282c34;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #a0a0a0;
            }
            QPushButton {
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #444;
                background-color: #2d313a;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #3a3f4b;
            }
            QPushButton:disabled {
                background-color: #1a1d23;
                color: #666;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QListWidget, QLabel {
                background-color: #23272e;
                color: #e0e0e0;
            }
            QComboBox:disabled {
                background-color: #1a1d23;
                color: #666;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 6px;
                background: #282c34;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #667eea;
                border: 1px solid #444;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QStatusBar {
                background: #23272e;
                color: #a0a0a0;
            }
            QMenuBar {
                background-color: #23272e;
                color: #e0e0e0;
            }
            QMenuBar::item:selected {
                background: #3a3f4b;
            }
            QMenu {
                background-color: #23272e;
                color: #e0e0e0;
            }
            QMenu::item:selected {
                background: #667eea;
                color: #fff;
            }
            QToolBar {
                background: #23272e;
                border-bottom: 1px solid #3a3f4b;
            }
            QSplitter::handle {
                background: #3a3f4b;
            }
        """)
        
        self.init_ui()
        self.create_menubar()
        self.create_toolbar()
        self.create_statusbar()
        self.connect_signals()
        self.load_settings()
        
    def init_ui(self):
        # Create central widget with main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tool panel (left)
        self.tool_panel = ToolPanel()
        self.tool_panel.setMinimumWidth(220)
        
        # Canvas (center)
        self.canvas = Canvas()
        self.canvas.setMinimumWidth(400)
        
        # Properties panel (right)
        self.properties_panel = PropertiesPanel()
        self.properties_panel.setMinimumWidth(180)
        
        # Layers panel (bottom right)
        self.layers_panel = LayersPanel()
        
        # Right side layout: properties panel on top, layers panel on bottom
        right_layout = QVBoxLayout()
        right_layout.setSpacing(0)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.properties_panel)
        right_layout.addWidget(self.layers_panel, stretch=2)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)
        splitter.addWidget(self.tool_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900, 400])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        # On startup, always show Background layer
        self.layers_panel._init_background_layer()
        
    def connect_signals(self):
        """Connect all signals and slots"""
        self.tool_panel.generate_cut_btn.clicked.connect(self.generate_cut_lines)
        self.tool_panel.remove_bg_btn.clicked.connect(self.remove_background)
        self.tool_panel.upscale_btn.clicked.connect(self.upscale_image)
        self.tool_panel.compare_btn.clicked.connect(self.show_upscale_comparison)
        self.tool_panel.offset_slider.valueChanged.connect(self.live_update_cut_lines)
        self.tool_panel.cut_type.currentIndexChanged.connect(self.live_update_cut_lines)
        self.tool_panel.offset_slider.valueChanged.connect(self.update_offset_label)
        self.tool_panel.smooth_slider.valueChanged.connect(self.update_smooth_label)
        self.tool_panel.smooth_slider.valueChanged.connect(self.live_update_cut_lines)
        self.layers_panel.list.itemChanged.connect(self.on_layer_visibility_changed)
        
    def create_menubar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Image/PDF", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save PDF", self)
        save_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Undo")
        edit_menu.addAction("Redo")
        edit_menu.addSeparator()
        edit_menu.addAction("Preferences")
        
        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Zoom In")
        view_menu.addAction("Zoom Out")
        fit_action = QAction("Fit to Window", self)
        fit_action.triggered.connect(self.canvas.fit_to_window)
        view_menu.addAction(fit_action)
        view_menu.addSeparator()
        view_menu.addAction("Show Cut Lines")
        view_menu.addAction("Show Original")
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("Batch Process")
        tools_menu.addAction("Presets Manager")
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("Documentation")
        help_menu.addAction("About")
        
    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Add toolbar actions with placeholder icons (replace with SVGs for branding)
        open_action = QAction(QIcon.fromTheme("document-open"), "Open", self)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)
        
        save_action = QAction(QIcon.fromTheme("document-save"), "Save", self)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # View tools
        zoom_in_action = QAction(QIcon.fromTheme("zoom-in"), "Zoom In", self)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction(QIcon.fromTheme("zoom-out"), "Zoom Out", self)
        toolbar.addAction(zoom_out_action)
        
        fit_action = QAction(QIcon.fromTheme("zoom-fit-best"), "Fit to Window", self)
        fit_action.triggered.connect(self.canvas.fit_to_window)
        toolbar.addAction(fit_action)
        
        toolbar.addSeparator()
        
        # Mutually exclusive show original/cutlines
        self.show_original_action = QAction("Show Original", self)
        self.show_original_action.setCheckable(True)
        self.show_cutlines_action = QAction("Show Cut Lines", self)
        self.show_cutlines_action.setCheckable(True)
        self.show_original_action.triggered.connect(self.toggle_show_original)
        self.show_cutlines_action.triggered.connect(self.toggle_show_cutlines)
        toolbar.addAction(self.show_original_action)
        toolbar.addAction(self.show_cutlines_action)
        self.show_original_action.setChecked(False)
        self.show_cutlines_action.setChecked(False)
        
    def create_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Add permanent widgets to status bar
        self.coords_label = QLabel("X: 0, Y: 0")
        self.zoom_label = QLabel("Zoom: 100%")
        self.gpu_label = QLabel("GPU: " + ("Available" if torch.cuda.is_available() else "CPU Mode"))
        
        self.statusbar.addPermanentWidget(self.coords_label)
        self.statusbar.addPermanentWidget(self.zoom_label)
        self.statusbar.addPermanentWidget(self.gpu_label)
        
        self.statusbar.showMessage("Ready")
        
    def toggle_show_original(self, checked):
        if checked:
            self.show_cutlines_action.setChecked(False)
            if hasattr(self, 'original_pixmap'):
                self.canvas.show_only_original(self.original_pixmap)
        else:
            self.show_processed_state()

    def toggle_show_cutlines(self, checked):
        if checked:
            self.show_original_action.setChecked(False)
            if hasattr(self, 'last_cutline_dict') and hasattr(self, 'last_color_dict'):
                self.canvas.show_only_cutlines(self.last_cutline_dict, self.last_color_dict)
        else:
            self.show_processed_state()

    def show_processed_state(self):
        if hasattr(self, 'processed_pixmap') and hasattr(self, 'last_cutline_dict') and hasattr(self, 'last_color_dict'):
            self.canvas.show_processed(self.processed_pixmap, self.last_cutline_dict, self.last_color_dict)

    def open_file(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Open Image or PDF", 
                "", 
                "Image Files (*.png *.jpg *.jpeg *.tiff *.bmp);;PDF Files (*.pdf);;All Files (*.*)"
            )
            
            if file_path:
                # Check if file exists
                if not os.path.exists(file_path):
                    QMessageBox.warning(self, "Error", "File not found")
                    return
                
                # Check file size
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                if file_size > 100:  # Warn for files over 100MB
                    reply = QMessageBox.question(self, "Large File", 
                        f"This file is {file_size:.1f}MB. Loading may be slow. Continue?",
                        QMessageBox.Yes | QMessageBox.No)
                    if reply != QMessageBox.Yes:
                        return
                
                # Try to load the image
                if not self.canvas.load_image(file_path):
                    QMessageBox.critical(self, "Error", "Failed to load image. File may be corrupted or in an unsupported format.")
                    return
                    
                self.statusbar.showMessage(f"Loaded: {os.path.basename(file_path)}")
                
                # Save original pixmap for 'Show Original'
                self.original_pixmap = QPixmap(file_path)
                self.processed_pixmap = QPixmap(file_path)
                
                # Reset upscaling state
                self.upscale_applied = False
                self.last_upscaled_result = None
                self.tool_panel.compare_btn.setEnabled(False)
                self.properties_panel.upscale_status.setText("Upscaling: Not applied")
                
                # Update properties panel
                self.update_image_properties(file_path)
                
                # Add image layer to layers panel
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    image = pixmap.toImage()
                    self.layers_panel.add_image_layer(os.path.basename(file_path), image)
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open file: {str(e)}")
            self.statusbar.showMessage("Error loading file")
    
    def update_image_properties(self, file_path):
        try:
            # Get actual image properties
            image = QImage(file_path)
            if not image.isNull():
                width = image.width()
                height = image.height()
                self.properties_panel.dimensions_label.setText(f"Dimensions: {width} x {height}")
                
                # Calculate DPI (assume 96 DPI if not specified)
                dpi_x = image.dotsPerMeterX() * 0.0254 if image.dotsPerMeterX() > 0 else 96
                dpi_y = image.dotsPerMeterY() * 0.0254 if image.dotsPerMeterY() > 0 else 96
                avg_dpi = int((dpi_x + dpi_y) / 2)
                self.properties_panel.resolution_label.setText(f"Resolution: {avg_dpi} DPI")
                
                # Determine color space
                if image.hasAlphaChannel():
                    colorspace = "RGBA"
                else:
                    colorspace = "RGB"
                self.properties_panel.colorspace_label.setText(f"Color Space: {colorspace}")
        except Exception as e:
            self.properties_panel.dimensions_label.setText("Dimensions: Error")
            self.properties_panel.resolution_label.setText("Resolution: Error")
            self.properties_panel.colorspace_label.setText("Color Space: Error")

    def qimage_to_numpy(self, qimage):
        """Safely convert QImage to numpy array"""
        try:
            width = qimage.width()
            height = qimage.height()
            bytes_per_line = qimage.bytesPerLine()
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bytes_per_line))
            
            # Handle different formats
            if qimage.format() == QImage.Format_RGB888:
                arr = arr[:, :width*3].reshape((height, width, 3))
            elif qimage.format() == QImage.Format_RGBA8888:
                arr = arr[:, :width*4].reshape((height, width, 4))
            else:
                # Convert to RGB888 if unknown format
                qimage = qimage.convertToFormat(QImage.Format_RGB888)
                return self.qimage_to_numpy(qimage)
                
            return arr
        except Exception as e:
            raise Exception(f"Failed to convert image: {str(e)}")

    def update_offset_label(self):
        offset_in = self.tool_panel.offset_slider.value() / 100.0
        self.tool_panel.offset_label.setText(f"{offset_in:.2f} in")

    def update_smooth_label(self):
        val = self.tool_panel.smooth_slider.value()
        self.tool_panel.smooth_label.setText(str(val))

    def live_update_cut_lines(self):
        # Only update if we already have cut lines generated
        if self.canvas.image_item and hasattr(self, 'last_cutline_dict'):
            self.generate_cut_lines()

    def upscale_image(self):
        """Run upscaling with Real-ESRGAN"""
        if not self.canvas.image_item:
            QMessageBox.warning(self, "No Image", "Please load an image before upscaling.")
            return
            
        # Get selected model name
        model_name = "RealESRGAN_x4plus"  # Default model
            
        # Get upscale factor from UI
        scale_text = self.tool_panel.upscale_factor.currentText()
        target_scale = int(scale_text[0])  # Extract number from "2x", "4x", etc.
            
        # Get current image as numpy array
        pixmap = self.canvas.image_item.pixmap()
        image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        arr = self.qimage_to_numpy(image)
            
        # Create progress dialog
        self.progress_dialog = QProgressDialog("Starting upscaling...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Upscaling Image")
        self.progress_dialog.setModal(True)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
            
        # Create and run upscaling thread
        self.upscaling_thread = UpscalingThread(arr, model_name, target_scale, tile_size=512)
        self.upscaling_thread.progress.connect(self.update_upscale_progress)
        self.upscaling_thread.finished.connect(self.on_upscale_complete)
        self.upscaling_thread.error.connect(self.on_upscale_error)
        self.upscaling_thread.start()

    def update_upscale_progress(self, value, text):
        if self.progress_dialog:
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(text)
            QApplication.processEvents()
    
    def on_upscale_complete(self, result_array, actual_scale):
        if self.progress_dialog:
            self.progress_dialog.close()
            
        try:
            # Convert result to QPixmap
            if result_array.shape[2] == 3:
                # RGB to RGBA
                rgba = np.zeros((result_array.shape[0], result_array.shape[1], 4), dtype=np.uint8)
                rgba[:, :, :3] = result_array
                rgba[:, :, 3] = 255
                result_array = rgba
                
            qimg = QImage(result_array.data, result_array.shape[1], result_array.shape[0], 
                         result_array.strides[0], QImage.Format_RGBA8888)
            upscaled_pixmap = QPixmap.fromImage(qimg)
            
            # Store the result
            self.last_upscaled_result = (upscaled_pixmap, actual_scale)
            
            # Show comparison dialog
            dialog = ComparisonDialog(self.processed_pixmap, upscaled_pixmap, actual_scale, self)
            if dialog.exec_() == QDialog.Accepted and dialog.use_upscaled:
                # Apply upscaling
                self.processed_pixmap = upscaled_pixmap
                self.canvas.update_image(upscaled_pixmap)
                self.upscale_applied = True
                
                # Update status
                self.statusbar.showMessage(f"Upscaling applied ({actual_scale:.2f}x)")
                self.properties_panel.upscale_status.setText(f"Upscaling: {actual_scale:.2f}x applied")
                self.properties_panel.dimensions_label.setText(f"Dimensions: {upscaled_pixmap.width()} x {upscaled_pixmap.height()}")
                
                # Add upscaled layer
                self.layers_panel.add_image_layer(f"Upscaled {actual_scale:.1f}x", qimg)
                
                # Enable compare button
                self.tool_panel.compare_btn.setEnabled(True)
            else:
                self.statusbar.showMessage("Upscaling cancelled - original kept")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply upscaling: {str(e)}")
            self.statusbar.showMessage("Error applying upscaling")
    
    def on_upscale_error(self, error_msg):
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, "Error", error_msg)
        self.statusbar.showMessage("Error during upscaling")
        
    def show_upscale_comparison(self):
        if self.last_upscaled_result:
            upscaled_pixmap, actual_scale = self.last_upscaled_result
            original = self.original_pixmap if not self.upscale_applied else self.processed_pixmap
            dialog = ComparisonDialog(original, upscaled_pixmap, actual_scale, self)
            dialog.exec_()

    def generate_cut_lines(self):
        if not self.canvas.image_item:
            self.statusbar.showMessage("No image loaded.")
            return
            
        # Cancel any existing thread
        if self.cutline_thread and self.cutline_thread.isRunning():
            self.cutline_thread.quit()
            self.cutline_thread.wait()
            
        self.canvas.clear_cut_lines()
        
        # Create progress dialog
        self.current_progress_dialog = QProgressDialog(self)
        self.current_progress_dialog.setWindowTitle("Processing")
        self.current_progress_dialog.setLabelText("Initializing...")
        self.current_progress_dialog.setCancelButton(None)
        self.current_progress_dialog.setMinimum(0)
        self.current_progress_dialog.setMaximum(100)
        self.current_progress_dialog.setWindowModality(Qt.WindowModal)
        self.current_progress_dialog.show()
        
        try:
            # Get image data
            pixmap = self.canvas.image_item.pixmap()
            image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
            arr = self.qimage_to_numpy(image)
            
            # Get settings
            offset_in = self.tool_panel.offset_slider.value() / 100.0
            dpi = 300
            offset_px = int(offset_in * dpi)
            if offset_px < 1:
                offset_px = 1
                
            smooth_val = self.tool_panel.smooth_slider.value()
            selected_type = self.tool_panel.cut_type.currentText()
            
            # Create and start thread
            self.cutline_thread = CutLineGenerationThread(arr, offset_px, smooth_val, selected_type)
            self.cutline_thread.progress.connect(self.update_cutline_progress)
            self.cutline_thread.finished.connect(self.on_cutlines_generated)
            self.cutline_thread.error.connect(self.on_cutline_error)
            self.cutline_thread.start()
            
        except Exception as e:
            self.current_progress_dialog.close()
            QMessageBox.critical(self, "Error", f"Failed to generate cut lines: {str(e)}")
            self.statusbar.showMessage("Error generating cut lines")
    
    def update_cutline_progress(self, value, text):
        if self.current_progress_dialog:
            self.current_progress_dialog.setValue(value)
            self.current_progress_dialog.setLabelText(text)
            QApplication.processEvents()
    
    def on_cutlines_generated(self, cutline_dict, color_dict):
        if self.current_progress_dialog:
            self.current_progress_dialog.close()
            
        self.last_cutline_dict = cutline_dict
        self.last_color_dict = color_dict
        self.processed_pixmap = self.canvas.image_item.pixmap()
        
        self.canvas.draw_cut_lines(cutline_dict, color_dict, width=2)
        
        # Update status
        cut_count = len(cutline_dict.get('CutContour', []))
        perf_count = len(cutline_dict.get('PerfCutContour', []))
        self.statusbar.showMessage(f"Generated {cut_count} CutContour and {perf_count} PerfCutContour cut lines.")
        self.properties_panel.cutline_status.setText(f"Cut Lines: {cut_count + perf_count} generated")
        
        # Add new cutline layer
        scene_rect = self.canvas.scene.sceneRect()
        selected_type = self.tool_panel.cut_type.currentText()
        
        if selected_type == 'CutContour' and cutline_dict['CutContour']:
            name = self.layers_panel.next_cutline_name('CutContour')
            self.layers_panel.add_cutline_layer(name, cutline_dict['CutContour'], scene_rect)
        if selected_type == 'PerfCutContour' and cutline_dict['PerfCutContour']:
            name = self.layers_panel.next_cutline_name('PerfCutContour')
            self.layers_panel.add_cutline_layer(name, cutline_dict['PerfCutContour'], scene_rect)
    
    def on_cutline_error(self, error_msg):
        if self.current_progress_dialog:
            self.current_progress_dialog.close()
        QMessageBox.critical(self, "Error", error_msg)
        self.statusbar.showMessage("Error generating cut lines")

    def remove_background(self):
        if not self.canvas.image_item:
            self.statusbar.showMessage("No image loaded.")
            return
            
        # Cancel any existing thread
        if self.bg_removal_thread and self.bg_removal_thread.isRunning():
            self.bg_removal_thread.quit()
            self.bg_removal_thread.wait()
            
        # Create progress dialog
        self.current_progress_dialog = QProgressDialog(self)
        self.current_progress_dialog.setWindowTitle("Background Removal")
        self.current_progress_dialog.setLabelText("Initializing...")
        self.current_progress_dialog.setCancelButton(None)
        self.current_progress_dialog.setMinimum(0)
        self.current_progress_dialog.setMaximum(100)
        self.current_progress_dialog.setWindowModality(Qt.WindowModal)
        self.current_progress_dialog.show()
        
        try:
            # Convert QPixmap to RGBA numpy array
            pixmap = self.canvas.image_item.pixmap()
            image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
            width, height = image.width(), image.height()
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))
            
            # Get alpha matting setting
            use_alpha_matting = self.tool_panel.alpha_matting.isChecked()
            
            # Create and start thread
            self.bg_removal_thread = BackgroundRemovalThread(arr, use_alpha_matting)
            self.bg_removal_thread.progress.connect(self.update_bg_progress)
            self.bg_removal_thread.finished.connect(self.on_bg_removed)
            self.bg_removal_thread.error.connect(self.on_bg_error)
            self.bg_removal_thread.start()
            
        except Exception as e:
            self.current_progress_dialog.close()
            QMessageBox.critical(self, "Error", f"Failed to start background removal: {str(e)}")
            self.statusbar.showMessage("Error removing background")
    
    def update_bg_progress(self, value, text):
        if self.current_progress_dialog:
            self.current_progress_dialog.setValue(value)
            self.current_progress_dialog.setLabelText(text)
            QApplication.processEvents()
    
    def on_bg_removed(self, result_array):
        if self.current_progress_dialog:
            self.current_progress_dialog.close()
            
        try:
            # Convert result back to QPixmap
            qimg = QImage(result_array.data, result_array.shape[1], result_array.shape[0], 
                         result_array.strides[0], QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimg)
            self.canvas.image_item.setPixmap(pixmap)
            self.processed_pixmap = pixmap
            
            # Update status
            self.statusbar.showMessage("Background removed successfully")
            self.properties_panel.bg_status.setText("Background: Removed (AI)")
            
            # Add new layer for background-removed image
            self.layers_panel.add_image_layer("Background Removed", qimg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply background removal: {str(e)}")
            self.statusbar.showMessage("Error applying background removal")
    
    def on_bg_error(self, error_msg):
        if self.current_progress_dialog:
            self.current_progress_dialog.close()
        QMessageBox.critical(self, "Error", error_msg)
        self.statusbar.showMessage("Error removing background")

    def on_layer_visibility_changed(self, item):
        if item:
            name = item.text()
            visible = item.checkState() == Qt.Checked
            self.canvas.show_cut_lines[name] = visible
            self.generate_cut_lines()
            
    def save_settings(self):
        """Save user preferences"""
        self.settings.setValue("offset", self.tool_panel.offset_slider.value())
        self.settings.setValue("smoothing", self.tool_panel.smooth_slider.value())
        self.settings.setValue("cutType", self.tool_panel.cut_type.currentIndex())
        self.settings.setValue("alphaMatting", self.tool_panel.alpha_matting.isChecked())
        self.settings.setValue("bgModel", self.tool_panel.bg_model.currentIndex())
        self.settings.setValue("upscaleModel", self.tool_panel.upscale_model.currentIndex())
        self.settings.setValue("upscaleFactor", self.tool_panel.upscale_factor.currentIndex())
        
    def load_settings(self):
        """Load user preferences"""
        offset = self.settings.value("offset", 6, type=int)
        self.tool_panel.offset_slider.setValue(offset)
        
        smoothing = self.settings.value("smoothing", 10, type=int)
        self.tool_panel.smooth_slider.setValue(smoothing)
        
        cut_type = self.settings.value("cutType", 0, type=int)
        self.tool_panel.cut_type.setCurrentIndex(cut_type)
        
        alpha_matting = self.settings.value("alphaMatting", True, type=bool)
        self.tool_panel.alpha_matting.setChecked(alpha_matting)
        
        bg_model = self.settings.value("bgModel", 0, type=int)
        self.tool_panel.bg_model.setCurrentIndex(bg_model)
        
        upscale_model = self.settings.value("upscaleModel", 0, type=int)
        self.tool_panel.upscale_model.setCurrentIndex(upscale_model)
        
        upscale_factor = self.settings.value("upscaleFactor", 2, type=int)
        self.tool_panel.upscale_factor.setCurrentIndex(upscale_factor)
        
    def closeEvent(self, event):
        """Save settings before closing"""
        self.save_settings()
        
        # Cancel any running threads
        if self.bg_removal_thread and self.bg_removal_thread.isRunning():
            self.bg_removal_thread.quit()
            self.bg_removal_thread.wait()
            
        if self.cutline_thread and self.cutline_thread.isRunning():
            self.cutline_thread.quit()
            self.cutline_thread.wait()
            
        if self.upscaling_thread and self.upscaling_thread.isRunning():
            self.upscaling_thread.quit()
            self.upscaling_thread.wait()
            
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look
    
    # Set application info for QSettings
    app.setOrganizationName("YourCompany")
    app.setApplicationName("RolandCutLineTool")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
