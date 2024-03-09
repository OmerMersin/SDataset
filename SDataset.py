from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QListWidget, QListWidgetItem, QDialog, QLabel, QScrollArea
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QIcon, QImage, QPainter, QColor
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QSize
import os
import cv2
import numpy as np
import json

detection = None


class ModelImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Importing")
        self.layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_area.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll_area)

        # Create layout for images and detections
        self.scroll_layout = QVBoxLayout(self.scroll_content)

        # Initialize dialog position to the left of the screen
        self.move(0, 0)

    def set_image(self, pixmap):
        # Clear previous images
        self.setWindowTitle("Model Detecting...")
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.takeAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Display new image
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        self.scroll_layout.addWidget(image_label)

        # Resize dialog to match image size
        self.resize(pixmap.size())

class DetectionThread(QThread):
    detection_finished = pyqtSignal(QPixmap, list)  # Signal emits file path, annotated image, and results

    def __init__(self, model, files, parent=None):
        super(DetectionThread, self).__init__(parent)
        self.model = model
        self.files = files
        self.annotations = []

    def run(self):
        for file_path in self.files:
            if os.path.isfile(file_path):  # If file is an image
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    frame = cv2.imread(file_path)
                    results = self.model(frame)[0]
                    annotated_image = self.annotate_image(frame, results)
                    self.detection_finished.emit(self.convert_cv_to_qpixmap(annotated_image), self.annotations)
                    self.save_annotations(file_path, results)
                elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                    cap = cv2.VideoCapture(file_path)
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        results = self.model(frame)[0]
                        annotated_image = self.annotate_image(frame, results)
                        self.detection_finished.emit(file_path, self.convert_cv_to_qpixmap(annotated_image), self.annotations)
                        self.save_annotations(file_path, results)
                    cap.release()

    def save_annotations(self, file_path, results):
        file_name = os.path.basename(file_path)
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, _, _ = result  # Assuming the result contains coordinates and dimensions
            self.annotations.append({
                "filename": file_name,
                "x": float(x1),
                "y": float(y1),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "class": ""
            })

    def annotate_image(self, image, results):
        annotated_image = image.copy()
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > 0.5:
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return annotated_image

    def convert_cv_to_qpixmap(self, image):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888).rgbSwapped()
        return QPixmap.fromImage(qImg)

class DragDropWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.layout = QVBoxLayout(self)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls() if os.path.isfile(u.toLocalFile())]
        if files:
            self.parent().add_images(files)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(127, 0, 0))  # Fill the background with white
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Drag Image Here")


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)

        # Left side
        self.drag_drop_widget = DragDropWidget()
        self.layout.addWidget(self.drag_drop_widget)

        # Right side
        self.file_browser = QListWidget()
        self.file_browser.setIconSize(QSize(200, 200))  # Set a larger icon size
        self.layout.addWidget(self.file_browser)

        # Select File or Directory button
        self.button = QPushButton("Select File or Directory")
        self.button.clicked.connect(self.open_file_or_directory)
        self.layout.addWidget(self.button)

        # Start Detection button
        self.start_detection_button = QPushButton("Start Detection")
        self.start_detection_button.clicked.connect(self.start_detection)
        self.layout.addWidget(self.start_detection_button)

        self.showMaximized()
        self.model = None
        self.model_import_dialog = None

    def open_file_or_directory(self):
        self.file_path = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if self.file_path:
            files = [os.path.join(self.file_path, f) for f in os.listdir(self.file_path) if os.path.isfile(os.path.join(self.file_path, f))]
            self.add_images(files)

    def add_images(self, files):
        for file_path in files:
            item = QListWidgetItem()
            pixmap = QPixmap(file_path).scaledToHeight(200)
            icon = QIcon(pixmap)  # Create QIcon from QPixmap
            item.setIcon(icon)  # Set QIcon as the icon for the list item
            item.setText(file_path)
            self.file_browser.addItem(item)

    def start_detection(self):
        global detection
        if not self.model:
            from ultralytics import YOLO
            model_path = 'yolov8x.pt'
            self.model = YOLO(model_path)

        self.model_import_dialog = ModelImportDialog()
        self.model_import_dialog.show()

        files = [self.file_browser.item(i).text() for i in range(self.file_browser.count())]
        if files:
            self.detection_thread = DetectionThread(self.model, files)
            self.detection_thread.detection_finished.connect(self.on_detection_finished)
            self.detection_thread.start()

    @pyqtSlot(QPixmap,list)
    def on_detection_finished(self, annotated_image, annotations):
        pixmap = annotated_image
        if self.model_import_dialog is not None:
            self.model_import_dialog.set_image(pixmap)

        with open('annotations.json', 'w') as json_file:
            json.dump(annotations, json_file, indent=4)
            json_file.write('\n')  # Add a newline after each annotation for better readability


if __name__ == '__main__':
    app = QApplication([])
    viewer = ImageViewer()
    viewer.show()
    app.exec()
