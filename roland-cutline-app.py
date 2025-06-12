import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QSlider, QGroupBox,
                           QToolBar, QAction, QFileDialog, QSplitter, QComboBox,
                           QSpinBox, QCheckBox, QTabWidget, QListWidget, QDockWidget,
                           QGraphicsScene, QGraphicsView, QGraphicsPixmapItem,
                           QStatusBar, QMenuBar, QMenu, QToolButton, QButtonGroup,
                           QRadioButton, QDoubleSpinBox, QFrame, QProgressBar, QToolTip)
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QRectF, QTimer, QSize
from PyQt5.QtGui import QPixmap, QPainter, QColor, QBrush, QPen, QIcon, QImage

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
        self.cut_lines = []
        self.current_zoom = 1.0
        self.show_cut_lines = True
        self.show_original = True
        
        # Background pattern - modern checkerboard for transparency
        self.setBackgroundBrush(QBrush(QColor("#f7fafc"), Qt.Dense6Pattern))
        
    def load_image(self, image_path):
        """Load an image onto the canvas"""
        pixmap = QPixmap(image_path)
        if self.image_item:
            self.scene.removeItem(self.image_item)
        
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
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
        upscale_label.setProperty("class", "header")
        ai_layout.addWidget(upscale_label)
        
        self.upscale_checkbox = QCheckBox("Enable Upscaling")
        self.upscale_checkbox.setToolTip("Enable AI-powered image upscaling to enhance resolution")
        ai_layout.addWidget(self.upscale_checkbox)
        
        self.upscale_factor = QComboBox()
        self.upscale_factor.addItems(["2x", "3x", "4x"])
        self.upscale_factor.setCurrentIndex(2)  # Default to 4x
        self.upscale_factor.setToolTip("Select upscaling factor - higher values provide better quality but take longer")
        ai_layout.addWidget(self.upscale_factor)
        
        self.upscale_progress = QProgressBar()
        self.upscale_progress.setVisible(False)
        self.upscale_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background-color: #f7fafc;
                text-align: center;
                font-size: 11px;
                color: #2d3748;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #667eea;
                border-radius: 5px;
                margin: 1px;
            }
        """)
        ai_layout.addWidget(self.upscale_progress)
        
        self.compare_btn = QPushButton("Compare Before/After")
        self.compare_btn.setEnabled(False)
        self.compare_btn.setToolTip("Compare original and upscaled images side by side")
        ai_layout.addWidget(self.compare_btn)
        
        ai_layout.addSpacing(10)
        
        # Background Removal
        bg_label = QLabel("Background Removal")
        bg_label.setProperty("class", "header")
        ai_layout.addWidget(bg_label)
        
        self.bg_model = QComboBox()
        self.bg_model.addItems(["U2-Net (Best Quality)", "U2-NetP (Faster)", "MODNet (Real-time)"])
        self.bg_model.setToolTip("Choose AI model: U2-Net for best quality, U2-NetP for balance, MODNet for speed")
        ai_layout.addWidget(self.bg_model)
        
        self.alpha_matting = QCheckBox("Enable Alpha Matting")
        self.alpha_matting.setChecked(True)
        self.alpha_matting.setToolTip("Enable alpha matting for smoother edge transitions and better quality")
        ai_layout.addWidget(self.alpha_matting)
        
        self.remove_bg_btn = QPushButton("Remove Background")
        self.remove_bg_btn.setToolTip("Remove image background using selected AI model")
        ai_layout.addWidget(self.remove_bg_btn)
        
        self.bg_progress = QProgressBar()
        self.bg_progress.setVisible(False)
        self.bg_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background-color: #f7fafc;
                text-align: center;
                font-size: 11px;
                color: #2d3748;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #667eea;
                border-radius: 5px;
                margin: 1px;
            }
        """)
        ai_layout.addWidget(self.bg_progress)
        
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        
        # Cut Line Settings
        cut_group = QGroupBox("Cut Line Settings")
        cut_layout = QVBoxLayout()
        
        # Offset setting
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Offset:"))
        self.offset_spinbox = QDoubleSpinBox()
        self.offset_spinbox.setRange(0.0, 1.0)
        self.offset_spinbox.setValue(0.06)
        self.offset_spinbox.setSingleStep(0.01)
        self.offset_spinbox.setSuffix(" in")
        self.offset_spinbox.setToolTip("Distance between image edge and cut line (0.06 inches recommended)")
        offset_layout.addWidget(self.offset_spinbox)
        cut_layout.addLayout(offset_layout)
        
        # Cut type selection
        self.cut_type = QComboBox()
        self.cut_type.addItems(["CutContour", "PerfCutContour"])
        self.cut_type.setToolTip("CutContour: Solid cut lines | PerfCutContour: Perforated cut lines for easy removal")
        cut_layout.addWidget(self.cut_type)
        
        # Generate button
        self.generate_cut_btn = QPushButton("Generate Cut Lines")
        self.generate_cut_btn.setProperty("class", "primary")
        self.generate_cut_btn.setToolTip("Generate precise cut lines for Roland cutting equipment")
        cut_layout.addWidget(self.generate_cut_btn)
        
        self.cutline_progress = QProgressBar()
        self.cutline_progress.setVisible(False)
        self.cutline_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background-color: #f7fafc;
                text-align: center;
                font-size: 11px;
                color: #2d3748;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #667eea;
                border-radius: 5px;
                margin: 1px;
            }
        """)
        cut_layout.addWidget(self.cutline_progress)
        
        cut_group.setLayout(cut_layout)
        layout.addWidget(cut_group)
        
        # Object Detection
        object_group = QGroupBox("Object Detection")
        object_layout = QVBoxLayout()
        
        self.detect_objects_btn = QPushButton("Detect Multiple Objects")
        self.detect_objects_btn.setToolTip("Automatically detect and identify multiple objects in the image")
        object_layout.addWidget(self.detect_objects_btn)
        
        self.object_progress = QProgressBar()
        self.object_progress.setVisible(False)
        self.object_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background-color: #f7fafc;
                text-align: center;
                font-size: 11px;
                color: #2d3748;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #667eea;
                border-radius: 5px;
                margin: 1px;
            }
        """)
        object_layout.addWidget(self.object_progress)
        
        self.object_list = QListWidget()
        self.object_list.setToolTip("List of detected objects - click to select and process individually")
        self.object_list.setMaximumHeight(100)
        object_layout.addWidget(self.object_list)
        
        object_group.setLayout(object_layout)
        layout.addWidget(object_group)
        
        # Export Settings
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        self.export_pdf_btn = QPushButton("Export PDF with Cut Lines")
        self.export_pdf_btn.setProperty("class", "success")
        self.export_pdf_btn.setToolTip("Export the processed image with cut lines as a PDF file")
        export_layout.addWidget(self.export_pdf_btn)
        
        self.export_progress = QProgressBar()
        self.export_progress.setVisible(False)
        self.export_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background-color: #f7fafc;
                text-align: center;
                font-size: 11px;
                color: #2d3748;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #48bb78;
                border-radius: 5px;
                margin: 1px;
            }
        """)
        export_layout.addWidget(self.export_progress)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        self.setLayout(layout)
        self.setMaximumWidth(340)
        self.setStyleSheet("""
            QWidget { 
                margin: 12px; 
                background-color: #f8fafc;
            }
            QWidget[class="panel"] {
                background-color: #ffffff;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
                padding: 16px;
                margin: 8px;
            }
        """)

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
        self.dimensions_label.setToolTip("Image width and height in pixels")
        self.resolution_label = QLabel("Resolution: -")
        self.resolution_label.setToolTip("Image resolution in DPI (dots per inch)")
        self.colorspace_label = QLabel("Color Space: -")
        self.colorspace_label.setToolTip("Image color space (RGB, CMYK, etc.)")
        
        img_layout.addWidget(self.dimensions_label)
        img_layout.addWidget(self.resolution_label)
        img_layout.addWidget(self.colorspace_label)
        
        img_group.setLayout(img_layout)
        layout.addWidget(img_group)
        
        # Processing Status
        status_group = QGroupBox("Processing Status")
        status_layout = QVBoxLayout()
        
        self.upscale_status = QLabel("Upscaling: Not applied")
        self.upscale_status.setToolTip("Current upscaling status and applied factor")
        self.bg_status = QLabel("Background: Original")
        self.bg_status.setToolTip("Background removal status and applied model")
        self.cutline_status = QLabel("Cut Lines: Not generated")
        self.cutline_status.setToolTip("Cut line generation status and parameters")
        
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
        self.edge_threshold.setToolTip("Adjust edge detection sensitivity - lower values detect more edges")
        self.edge_label = QLabel("Edge Sensitivity: 50")
        self.edge_label.setToolTip("Current edge detection sensitivity setting")
        
        advanced_layout.addWidget(self.edge_label)
        advanced_layout.addWidget(self.edge_threshold)
        
        self.edge_threshold.valueChanged.connect(
            lambda v: self.edge_label.setText(f"Edge Sensitivity: {v}")
        )
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        self.setLayout(layout)
        self.setMaximumWidth(290)
        self.setStyleSheet("""
            QWidget { 
                margin: 12px; 
                background-color: #f8fafc;
            }
            QWidget[class="panel"] {
                background-color: #ffffff;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
                padding: 16px;
                margin: 8px;
            }
        """)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Roland Cut Line Tool - Professional Edition")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application style with enhanced modern theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f1f5f9;
                color: #2d3748;
                border: none;
            }
            
            /* Panel Styling */
            QWidget {
                background-color: #f1f5f9;
                color: #2d3748;
            }
            
            /* Main Content Area Styling */
            QSplitter {
                background-color: #f1f5f9;
                border: none;
            }
            QSplitter::handle {
                background-color: #e2e8f0;
                border: 1px solid #cbd5e0;
                border-radius: 2px;
                margin: 2px;
            }
            QSplitter::handle:hover {
                background-color: #cbd5e0;
            }
            
            /* Group Box Styling */
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #2d3748;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                margin: 8px;
                margin-top: 24px;
                padding-top: 24px;
                padding-left: 16px;
                padding-right: 16px;
                padding-bottom: 16px;
                background-color: #ffffff;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07), 0 1px 3px rgba(0, 0, 0, 0.06);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 4px 12px 4px 12px;
                background-color: #ffffff;
                color: #2d3748;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            }
            
            /* Button Styling */
            QPushButton {
                padding: 12px 20px;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
                background-color: #ffffff;
                color: #2d3748;
                font-weight: 500;
                font-size: 13px;
                min-height: 16px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }
            QPushButton:hover {
                background-color: #f7fafc;
                border-color: #cbd5e0;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
                transform: translateY(-1px);
            }
            QPushButton:pressed {
                background-color: #edf2f7;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                transform: translateY(0px);
            }
            QPushButton:disabled {
                background-color: #f7fafc;
                color: #a0aec0;
                border-color: #e2e8f0;
                box-shadow: none;
            }
            
            /* Primary Action Buttons */
            QPushButton[class="primary"] {
                background-color: #667eea;
                color: white;
                border: 1px solid #667eea;
                font-weight: 600;
                box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
            }
            QPushButton[class="primary"]:hover {
                background-color: #5a67d8;
                border-color: #5a67d8;
                box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
                transform: translateY(-2px);
            }
            QPushButton[class="primary"]:pressed {
                background-color: #4c51bf;
                box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
                transform: translateY(0px);
            }
            
            /* Success Action Buttons */
            QPushButton[class="success"] {
                background-color: #48bb78;
                color: white;
                border: 1px solid #48bb78;
                font-weight: 600;
                box-shadow: 0 2px 4px rgba(72, 187, 120, 0.3);
            }
            QPushButton[class="success"]:hover {
                background-color: #38a169;
                border-color: #38a169;
                box-shadow: 0 4px 8px rgba(72, 187, 120, 0.4);
                transform: translateY(-2px);
            }
            QPushButton[class="success"]:pressed {
                background-color: #2f855a;
                box-shadow: 0 2px 4px rgba(72, 187, 120, 0.3);
                transform: translateY(0px);
            }
            
            /* Form Controls */
            QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 8px 12px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background-color: #ffffff;
                color: #2d3748;
                font-size: 13px;
                min-height: 20px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #cbd5e0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #667eea;
                outline: none;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            /* Checkbox Styling */
            QCheckBox {
                color: #2d3748;
                font-size: 13px;
                spacing: 10px;
                font-weight: 500;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #e2e8f0;
                background-color: #ffffff;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            }
            QCheckBox::indicator:hover {
                border-color: #cbd5e0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            QCheckBox::indicator:checked {
                background-color: #667eea;
                border-color: #667eea;
                box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
            }
            QCheckBox::indicator:checked:hover {
                background-color: #5a67d8;
                border-color: #5a67d8;
            }
            
            /* Label Styling */
            QLabel {
                color: #2d3748;
                font-size: 12px;
            }
            QLabel[class="header"] {
                font-weight: bold;
                font-size: 14px;
                color: #2d3748;
            }
            QLabel[class="secondary"] {
                color: #4a5568;
                font-size: 11px;
            }
            
            /* Slider Styling */
            QSlider::groove:horizontal {
                border: 1px solid #e2e8f0;
                height: 4px;
                background: #edf2f7;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #667eea;
                border: 1px solid #667eea;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #5a67d8;
                border-color: #5a67d8;
            }
            
            /* List Widget */
            QListWidget {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background-color: #ffffff;
                color: #2d3748;
                font-size: 13px;
                padding: 6px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }
            QListWidget::item {
                padding: 8px 12px;
                border-radius: 6px;
                margin: 2px;
                font-weight: 500;
            }
            QListWidget::item:hover {
                background-color: #f7fafc;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            }
            QListWidget::item:selected {
                background-color: #667eea;
                color: white;
                box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
            }
            
            /* Toolbar Styling */
            QToolBar {
                background-color: #ffffff;
                border: none;
                border-bottom: 1px solid #e2e8f0;
                padding: 8px 12px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                spacing: 4px;
            }
            QToolBar QToolButton {
                padding: 8px 12px;
                border-radius: 6px;
                border: 1px solid transparent;
                background-color: transparent;
                color: #2d3748;
                font-weight: 500;
                margin: 2px;
            }
            QToolBar QToolButton:hover {
                background-color: #f7fafc;
                border-color: #e2e8f0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            QToolBar QToolButton:pressed {
                background-color: #edf2f7;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            }
            
            /* Status Bar */
            QStatusBar {
                background-color: #ffffff;
                border-top: 1px solid #e2e8f0;
                color: #4a5568;
                font-size: 11px;
                padding: 6px 12px;
                box-shadow: 0 -1px 3px rgba(0, 0, 0, 0.1);
            }
            
            /* Menu Bar */
            QMenuBar {
                background-color: #ffffff;
                border-bottom: 1px solid #e2e8f0;
                color: #2d3748;
                font-size: 12px;
                padding: 4px 8px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }
            QMenuBar::item {
                padding: 8px 16px;
                background-color: transparent;
                border-radius: 6px;
                margin: 2px;
                font-weight: 500;
            }
            QMenuBar::item:hover {
                background-color: #f7fafc;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            QMenuBar::item:pressed {
                background-color: #edf2f7;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            }
            
            /* Menu Dropdown Styling */
            QMenu {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 6px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
            }
            QMenu::item {
                padding: 8px 16px;
                border-radius: 6px;
                color: #2d3748;
                font-size: 12px;
            }
            QMenu::item:hover {
                background-color: #f7fafc;
                color: #2d3748;
            }
            QMenu::item:selected {
                background-color: #667eea;
                color: white;
            }
        """)
        
        self.init_ui()
        self.create_menubar()
        self.create_toolbar()
        self.create_statusbar()
        
        self.connect_processing_buttons()
        
    def init_ui(self):
        # Create central widget with main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        
        # Tool panel (left)
        self.tool_panel = ToolPanel()
        
        # Canvas (center)
        self.canvas = Canvas()
        
        # Properties panel (right)
        self.properties_panel = PropertiesPanel()
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tool_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.properties_panel)
        
        # Set initial splitter sizes with enhanced proportions
        splitter.setSizes([340, 800, 290])
        
        splitter.setCollapsible(0, False)  # Tool panel cannot be collapsed
        splitter.setCollapsible(1, False)  # Canvas cannot be collapsed
        splitter.setCollapsible(2, False)  # Properties panel cannot be collapsed
        
        self.tool_panel.setMinimumWidth(280)
        self.canvas.setMinimumSize(400, 300)  # Minimum canvas size
        self.properties_panel.setMinimumWidth(220)
        
        self.main_splitter = splitter
        
        splitter.splitterMoved.connect(self.on_splitter_moved)
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
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
        
        # View menu with enhanced canvas controls
        view_menu = menubar.addMenu("View")
        
        zoom_in_menu = QAction("Zoom In", self)
        zoom_in_menu.setShortcut("Ctrl++")
        zoom_in_menu.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_menu)
        
        zoom_out_menu = QAction("Zoom Out", self)
        zoom_out_menu.setShortcut("Ctrl+-")
        zoom_out_menu.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_menu)
        
        fit_menu = QAction("Fit to Window", self)
        fit_menu.setShortcut("Ctrl+0")
        fit_menu.triggered.connect(self.fit_to_window)
        view_menu.addAction(fit_menu)
        
        actual_size_menu = QAction("Actual Size", self)
        actual_size_menu.setShortcut("Ctrl+1")
        actual_size_menu.triggered.connect(self.actual_size)
        view_menu.addAction(actual_size_menu)
        
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
        
        # Add toolbar actions
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)
        
        save_action = QAction("Save", self)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # View tools with enhanced canvas scaling
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)
        
        fit_action = QAction("Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.fit_to_window)
        toolbar.addAction(fit_action)
        
        actual_size_action = QAction("Actual Size", self)
        actual_size_action.setShortcut("Ctrl+1")
        actual_size_action.triggered.connect(self.actual_size)
        toolbar.addAction(actual_size_action)
        
        toolbar.addSeparator()
        
        # Toggle buttons
        toggle_original = QAction("Show Original", self)
        toggle_original.setCheckable(True)
        toggle_original.setChecked(True)
        toolbar.addAction(toggle_original)
        
        toggle_cutlines = QAction("Show Cut Lines", self)
        toggle_cutlines.setCheckable(True)
        toggle_cutlines.setChecked(True)
        toolbar.addAction(toggle_cutlines)
        
    def create_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Add permanent widgets to status bar
        self.coords_label = QLabel("X: 0, Y: 0")
        self.coords_label.setToolTip("Current mouse coordinates on canvas")
        
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setToolTip("Current canvas zoom level")
        
        # Processing status indicator
        self.processing_label = QLabel("")
        self.processing_label.setStyleSheet("color: #667eea; font-weight: bold;")
        
        self.memory_label = QLabel("Memory: 0 MB")
        self.memory_label.setToolTip("Current memory usage")
        self.memory_label.setStyleSheet("color: #4a5568; font-size: 11px;")
        
        self.statusbar.addWidget(self.processing_label)
        self.statusbar.addPermanentWidget(self.memory_label)
        self.statusbar.addPermanentWidget(self.coords_label)
        self.statusbar.addPermanentWidget(self.zoom_label)
        
        self.statusbar.showMessage("Ready - Load an image to begin processing")
        
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image or PDF", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.tiff *.bmp);;PDF Files (*.pdf);;All Files (*.*)"
        )
        
        if file_path:
            self.processing_label.setText("Loading image...")
            self.statusbar.showMessage("Loading image, please wait...")
            
            QTimer.singleShot(100, lambda: self._load_image_delayed(file_path))
    
    def _load_image_delayed(self, file_path):
        """Load image with progress feedback"""
        try:
            self.canvas.load_image(file_path)
            
            self.processing_label.setText("")
            self.statusbar.showMessage(f"✓ Loaded: {os.path.basename(file_path)} - Ready for processing")
            
            # Update properties panel
            self.update_image_properties(file_path)
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            self.memory_label.setText(f"Memory: {file_size:.1f} MB")
            
        except Exception as e:
            self.processing_label.setText("")
            self.statusbar.showMessage(f"✗ Error loading file: {str(e)}")
            self.show_error_tooltip(f"Failed to load image: {str(e)}")
    
    def show_error_tooltip(self, message):
        """Show error message as tooltip"""
        QToolTip.showText(self.mapToGlobal(self.rect().center()), message)
    
    def show_success_tooltip(self, message):
        """Show success message as tooltip"""
        QToolTip.showText(self.mapToGlobal(self.rect().center()), message)
    
    def on_splitter_moved(self):
        """Handle splitter movement for responsive canvas behavior"""
        if hasattr(self.canvas, 'pixmap_item') and self.canvas.pixmap_item:
            current_size = self.canvas.size()
            if not hasattr(self, '_last_canvas_size'):
                self._last_canvas_size = current_size
                return
            
            size_change = abs(current_size.width() - self._last_canvas_size.width()) / max(self._last_canvas_size.width(), 1)
            if size_change > 0.2:  # 20% change threshold
                self.fit_to_window()
                self._last_canvas_size = current_size
    
    def fit_to_window(self):
        """Fit image to window size"""
        if hasattr(self.canvas, 'pixmap_item') and self.canvas.pixmap_item:
            self.canvas.fitInView(self.canvas.pixmap_item, Qt.KeepAspectRatio)
            self.update_zoom_status()
    
    def update_zoom_status(self):
        """Update zoom level in status bar"""
        if hasattr(self.canvas, 'transform'):
            transform = self.canvas.transform()
            zoom_level = transform.m11() * 100  # Get scale factor as percentage
            self.zoom_label.setText(f"Zoom: {zoom_level:.0f}%")
    
    def zoom_in(self):
        """Zoom in on the canvas"""
        if hasattr(self.canvas, 'scale'):
            self.canvas.scale(1.25, 1.25)
            self.update_zoom_status()
    
    def zoom_out(self):
        """Zoom out on the canvas"""
        if hasattr(self.canvas, 'scale'):
            self.canvas.scale(0.8, 0.8)
            self.update_zoom_status()
    
    def actual_size(self):
        """Show image at actual size"""
        if hasattr(self.canvas, 'resetTransform'):
            self.canvas.resetTransform()
            self.update_zoom_status()
    
    def resizeEvent(self, event):
        """Handle window resize events with responsive panel behavior"""
        super().resizeEvent(event)
        
        window_width = event.size().width()
        
        if window_width < 1200:  # Compact mode
            self.tool_panel.setMaximumWidth(300)
            self.properties_panel.setMaximumWidth(250)
        elif window_width < 1600:  # Standard mode
            self.tool_panel.setMaximumWidth(340)
            self.properties_panel.setMaximumWidth(290)
        else:  # Expanded mode
            self.tool_panel.setMaximumWidth(380)
            self.properties_panel.setMaximumWidth(320)
        
        if hasattr(self, 'canvas') and hasattr(self.canvas, 'pixmap_item') and self.canvas.pixmap_item:
            QTimer.singleShot(100, self.fit_to_window)  # Delay to ensure layout is updated
    
    def update_image_properties(self, file_path):
        # This would be connected to actual image analysis
        self.properties_panel.dimensions_label.setText("Dimensions: 1920 x 1080")
        self.properties_panel.resolution_label.setText("Resolution: 300 DPI")
        self.properties_panel.colorspace_label.setText("Color Space: RGB")
    
    def simulate_upscaling(self):
        """Simulate upscaling process with progress feedback"""
        self.tool_panel.upscale_progress.setVisible(True)
        self.tool_panel.upscale_progress.setValue(0)
        self.processing_label.setText("Upscaling image...")
        self.statusbar.showMessage("Upscaling in progress...")
        
        self.upscale_timer = QTimer()
        self.upscale_progress_value = 0
        self.upscale_timer.timeout.connect(self.update_upscale_progress)
        self.upscale_timer.start(100)  # Update every 100ms
    
    def update_upscale_progress(self):
        """Update upscaling progress"""
        self.upscale_progress_value += 2
        self.tool_panel.upscale_progress.setValue(self.upscale_progress_value)
        
        if self.upscale_progress_value >= 100:
            self.upscale_timer.stop()
            self.tool_panel.upscale_progress.setVisible(False)
            self.processing_label.setText("")
            self.statusbar.showMessage("✓ Upscaling completed successfully")
            self.show_success_tooltip("Image upscaling completed!")
            
            # Update properties panel
            factor = self.tool_panel.upscale_factor.currentText()
            self.properties_panel.upscale_status.setText(f"Upscaling: Applied {factor}")
    
    def simulate_background_removal(self):
        """Simulate background removal process with progress feedback"""
        self.tool_panel.bg_progress.setVisible(True)
        self.tool_panel.bg_progress.setValue(0)
        self.processing_label.setText("Removing background...")
        self.statusbar.showMessage("Background removal in progress...")
        
        self.bg_timer = QTimer()
        self.bg_progress_value = 0
        self.bg_timer.timeout.connect(self.update_bg_progress)
        self.bg_timer.start(150)  # Update every 150ms
    
    def update_bg_progress(self):
        """Update background removal progress"""
        self.bg_progress_value += 3
        self.tool_panel.bg_progress.setValue(self.bg_progress_value)
        
        if self.bg_progress_value >= 100:
            self.bg_timer.stop()
            self.tool_panel.bg_progress.setVisible(False)
            self.processing_label.setText("")
            self.statusbar.showMessage("✓ Background removal completed successfully")
            self.show_success_tooltip("Background removed successfully!")
            
            # Update properties panel
            model = self.tool_panel.bg_model.currentText()
            self.properties_panel.bg_status.setText(f"Background: Removed ({model})")
    
    def simulate_cutline_generation(self):
        """Simulate cut line generation process with progress feedback"""
        self.tool_panel.cutline_progress.setVisible(True)
        self.tool_panel.cutline_progress.setValue(0)
        self.processing_label.setText("Generating cut lines...")
        self.statusbar.showMessage("Cut line generation in progress...")
        
        self.cutline_timer = QTimer()
        self.cutline_progress_value = 0
        self.cutline_timer.timeout.connect(self.update_cutline_progress)
        self.cutline_timer.start(80)  # Update every 80ms
    
    def update_cutline_progress(self):
        """Update cut line generation progress"""
        self.cutline_progress_value += 4
        self.tool_panel.cutline_progress.setValue(self.cutline_progress_value)
        
        if self.cutline_progress_value >= 100:
            self.cutline_timer.stop()
            self.tool_panel.cutline_progress.setVisible(False)
            self.processing_label.setText("")
            self.statusbar.showMessage("✓ Cut lines generated successfully")
            self.show_success_tooltip("Cut lines generated successfully!")
            
            # Update properties panel
            cut_type = self.tool_panel.cut_type.currentText()
            offset = self.tool_panel.offset_spinbox.value()
            self.properties_panel.cutline_status.setText(f"Cut Lines: Generated ({cut_type}, {offset}in offset)")
    
    def simulate_object_detection(self):
        """Simulate object detection process with progress feedback"""
        self.tool_panel.object_progress.setVisible(True)
        self.tool_panel.object_progress.setValue(0)
        self.processing_label.setText("Detecting objects...")
        self.statusbar.showMessage("Object detection in progress...")
        
        self.object_timer = QTimer()
        self.object_progress_value = 0
        self.object_timer.timeout.connect(self.update_object_progress)
        self.object_timer.start(120)  # Update every 120ms
    
    def update_object_progress(self):
        """Update object detection progress"""
        self.object_progress_value += 5
        self.tool_panel.object_progress.setValue(self.object_progress_value)
        
        if self.object_progress_value >= 100:
            self.object_timer.stop()
            self.tool_panel.object_progress.setVisible(False)
            self.processing_label.setText("")
            self.statusbar.showMessage("✓ Object detection completed successfully")
            self.show_success_tooltip("Objects detected successfully!")
            
            self.tool_panel.object_list.clear()
            self.tool_panel.object_list.addItems(["Object 1: Logo", "Object 2: Text", "Object 3: Shape"])
    
    def simulate_export(self):
        """Simulate export process with progress feedback"""
        self.tool_panel.export_progress.setVisible(True)
        self.tool_panel.export_progress.setValue(0)
        self.processing_label.setText("Exporting PDF...")
        self.statusbar.showMessage("PDF export in progress...")
        
        self.export_timer = QTimer()
        self.export_progress_value = 0
        self.export_timer.timeout.connect(self.update_export_progress)
        self.export_timer.start(90)  # Update every 90ms
    
    def update_export_progress(self):
        """Update export progress"""
        self.export_progress_value += 3
        self.tool_panel.export_progress.setValue(self.export_progress_value)
        
        if self.export_progress_value >= 100:
            self.export_timer.stop()
            self.tool_panel.export_progress.setVisible(False)
            self.processing_label.setText("")
            self.statusbar.showMessage("✓ PDF exported successfully")
            self.show_success_tooltip("PDF exported successfully!")
    
    def connect_processing_buttons(self):
        """Connect processing buttons to simulated operations"""
        if hasattr(self.tool_panel, 'upscale_checkbox'):
            self.tool_panel.upscale_checkbox.stateChanged.connect(
                lambda: self.simulate_upscaling() if self.tool_panel.upscale_checkbox.isChecked() else None
            )
        
        self.tool_panel.remove_bg_btn.clicked.connect(self.simulate_background_removal)
        
        self.tool_panel.generate_cut_btn.clicked.connect(self.simulate_cutline_generation)
        
        self.tool_panel.detect_objects_btn.clicked.connect(self.simulate_object_detection)
        
        self.tool_panel.export_pdf_btn.clicked.connect(self.simulate_export)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
