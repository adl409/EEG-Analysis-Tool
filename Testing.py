import sys
import os
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QMenu, QVBoxLayout, QWidget, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QMainWindow, QTabWidget, QWidget, QSizePolicy, QGroupBox, QComboBox, QTableView, QFrame
from PySide6.QtCore import QRegularExpression, Signal, QObject, QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QRegularExpressionValidator
import sys
import tensorflow as tf


class ConfigMenu(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = None  # Placeholder for the model

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Testing Configuration Menu')
        self.setGeometry(400,400,400,400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Button to load the model
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.loadModel)
        self.layout.addWidget(self.load_model_button)

        # Label to display model loading status
        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

    def loadModel(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Model files (*.h5 *.hdf5 *.pb *.pbtxt)")
        file_dialog.setWindowTitle("Select Model File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                model_path = selected_files[0]
                try:
                    self.model = tf.keras.models.load_model(model_path)
                    self.status_label.setText("Model loaded successfully")
                except Exception as e:
                    self.status_label.setText("Failed to load model")

def main():
    app = QApplication(sys.argv)
    ex = ConfigMenu()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()




