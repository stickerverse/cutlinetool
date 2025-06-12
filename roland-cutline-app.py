import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QSlider, QGroupBox,
                           QToolBar, QAction, QFileDialog, QSplitter, QComboBox,
                           QSpinBox, QCheckBox, QTabWidget, QListWidget, QDockWidget,
                           QGraphicsScene, QGraphicsView, QGraphicsPixmapItem,
                           QStatusBar, QMenuBar, QMenu, QToolButton, QButtonGroup,
                           QRadioButton, QDoubleSpinBox, QFrame)
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QRectF, QTimer
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
        ai_layout.addWidget(self.upscale_checkbox)
        
        self.upscale_factor = QComboBox()
        self.upscale_factor.addItems(["2x", "3x", "4x"])
        self.upscale_factor.setCurrentIndex(2)  # Default to 4x
        ai_layout.addWidget(self.upscale_factor)
        
        self.compare_btn = QPushButton("Compare Before/After")
        self.compare_btn.setEnabled(False)
        ai_layout.addWidget(self.compare_btn)
        
        ai_layout.addSpacing(10)
        
        # Background Removal
        bg_label = QLabel("Background Removal")
        bg_label.setProperty("class", "header")
        ai_layout.addWidget(bg_label)
        
        self.bg_model = QComboBox()
        self.bg_model.addItems(["U2-Net (Best Quality)", "U2-NetP (Faster)", "MODNet (Real-time)"])
        ai_layout.addWidget(self.bg_model)
        
        self.alpha_matting = QCheckBox("Enable Alpha Matting")
        self.alpha_matting.setChecked(True)
        ai_layout.addWidget(self.alpha_matting)
        
        self.remove_bg_btn = QPushButton("Remove Background")
        ai_layout.addWidget(self.remove_bg_btn)
        
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
        offset_layout.addWidget(self.offset_spinbox)
        cut_layout.addLayout(offset_layout)
        
        # Cut type selection
        self.cut_type = QComboBox()
        self.cut_type.addItems(["CutContour", "PerfCutContour"])
        cut_layout.addWidget(self.cut_type)
        
        # Generate button
        self.generate_cut_btn = QPushButton("Generate Cut Lines")
        self.generate_cut_btn.setProperty("class", "primary")
        cut_layout.addWidget(self.generate_cut_btn)
        
        cut_group.setLayout(cut_layout)
        layout.addWidget(cut_group)
        
        # Object Detection
        object_group = QGroupBox("Object Detection")
        object_layout = QVBoxLayout()
        
        self.detect_objects_btn = QPushButton("Detect Multiple Objects")
        object_layout.addWidget(self.detect_objects_btn)
        
        self.object_list = QListWidget()
        self.object_list.setMaximumHeight(100)
        object_layout.addWidget(self.object_list)
        
        object_group.setLayout(object_layout)
        layout.addWidget(object_group)
        
        # Export Settings
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        self.export_pdf_btn = QPushButton("Export PDF with Cut Lines")
        self.export_pdf_btn.setProperty("class", "success")
        export_layout.addWidget(self.export_pdf_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        self.setLayout(layout)
        self.setMaximumWidth(300)

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
        
        # Set application style with enhanced modern theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
                color: #2d3748;
            }
            
            /* Panel Styling */
            QWidget {
                background-color: #f8f9fa;
                color: #2d3748;
            }
            
            /* Group Box Styling */
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #2d3748;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                background-color: #ffffff;
                color: #2d3748;
            }
            
            /* Button Styling */
            QPushButton {
                padding: 10px 16px;
                border-radius: 6px;
                border: 1px solid #e2e8f0;
                background-color: #ffffff;
                color: #2d3748;
                font-weight: 500;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #f7fafc;
                border-color: #cbd5e0;
            }
            QPushButton:pressed {
                background-color: #edf2f7;
            }
            QPushButton:disabled {
                background-color: #f7fafc;
                color: #a0aec0;
                border-color: #e2e8f0;
            }
            
            /* Primary Action Buttons */
            QPushButton[class="primary"] {
                background-color: #667eea;
                color: white;
                border: 1px solid #667eea;
            }
            QPushButton[class="primary"]:hover {
                background-color: #5a67d8;
                border-color: #5a67d8;
            }
            QPushButton[class="primary"]:pressed {
                background-color: #4c51bf;
            }
            
            /* Success Action Buttons */
            QPushButton[class="success"] {
                background-color: #48bb78;
                color: white;
                border: 1px solid #48bb78;
            }
            QPushButton[class="success"]:hover {
                background-color: #38a169;
                border-color: #38a169;
            }
            QPushButton[class="success"]:pressed {
                background-color: #2f855a;
            }
            
            /* Form Controls */
            QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 6px 10px;
                border: 1px solid #e2e8f0;
                border-radius: 4px;
                background-color: #ffffff;
                color: #2d3748;
                font-size: 12px;
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #cbd5e0;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #667eea;
                outline: none;
            }
            
            /* Checkbox Styling */
            QCheckBox {
                color: #2d3748;
                font-size: 12px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #e2e8f0;
                background-color: #ffffff;
            }
            QCheckBox::indicator:hover {
                border-color: #cbd5e0;
            }
            QCheckBox::indicator:checked {
                background-color: #667eea;
                border-color: #667eea;
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
                border-radius: 4px;
                background-color: #ffffff;
                color: #2d3748;
                font-size: 12px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-radius: 3px;
            }
            QListWidget::item:hover {
                background-color: #f7fafc;
            }
            QListWidget::item:selected {
                background-color: #667eea;
                color: white;
            }
            
            /* Toolbar Styling */
            QToolBar {
                background-color: #ffffff;
                border: none;
                border-bottom: 1px solid #e2e8f0;
                padding: 4px;
            }
            QToolBar QToolButton {
                padding: 6px;
                border-radius: 4px;
                border: none;
                background-color: transparent;
                color: #2d3748;
            }
            QToolBar QToolButton:hover {
                background-color: #f7fafc;
            }
            QToolBar QToolButton:pressed {
                background-color: #edf2f7;
            }
            
            /* Status Bar */
            QStatusBar {
                background-color: #ffffff;
                border-top: 1px solid #e2e8f0;
                color: #4a5568;
                font-size: 11px;
            }
            
            /* Menu Bar */
            QMenuBar {
                background-color: #ffffff;
                border-bottom: 1px solid #e2e8f0;
                color: #2d3748;
                font-size: 12px;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background-color: transparent;
            }
            QMenuBar::item:hover {
                background-color: #f7fafc;
            }
            QMenuBar::item:pressed {
                background-color: #edf2f7;
            }
        """)
        
        self.init_ui()
        self.create_menubar()
        self.create_toolbar()
        self.create_statusbar()
        
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
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tool_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.properties_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([300, 800, 300])
        
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
        
        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Zoom In")
        view_menu.addAction("Zoom Out")
        view_menu.addAction("Fit to Window")
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
        toolbar.setIconSize(Qt.QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Add toolbar actions
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)
        
        save_action = QAction("Save", self)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # View tools
        zoom_in_action = QAction("Zoom In", self)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        toolbar.addAction(zoom_out_action)
        
        fit_action = QAction("Fit to Window", self)
        toolbar.addAction(fit_action)
        
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
        self.zoom_label = QLabel("Zoom: 100%")
        
        self.statusbar.addPermanentWidget(self.coords_label)
        self.statusbar.addPermanentWidget(self.zoom_label)
        
        self.statusbar.showMessage("Ready")
        
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image or PDF", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.tiff *.bmp);;PDF Files (*.pdf);;All Files (*.*)"
        )
        
        if file_path:
            self.canvas.load_image(file_path)
            self.statusbar.showMessage(f"Loaded: {os.path.basename(file_path)}")
            
            # Update properties panel
            self.update_image_properties(file_path)
    
    def update_image_properties(self, file_path):
        # This would be connected to actual image analysis
        self.properties_panel.dimensions_label.setText("Dimensions: 1920 x 1080")
        self.properties_panel.resolution_label.setText("Resolution: 300 DPI")
        self.properties_panel.colorspace_label.setText("Color Space: RGB")

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
