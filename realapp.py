import sys
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QMainWindow, QTabWidget, QWidget, QSizePolicy, QGroupBox, QComboBox
from PySide6.QtCore import QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator
import os

class ConfigureAddLayerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Layer")

        # Create a main layout
        layout = QVBoxLayout(self)

        # Create a container widget and layout for combo boxes and line edits
        container = QWidget()
        container_layout = QHBoxLayout(container)
        layout.addWidget(container)

        # Left side for dropdowns
        left_layout = QVBoxLayout()
        for _ in range(4):
            comboBox = QComboBox()
            comboBox.addItems(["Option 1", "Option 2", "Option 3"])  # Placeholder options
            left_layout.addWidget(comboBox)

        # Right side for line edits
        right_layout = QVBoxLayout()

        #label for lineedits
        #input_field_label = QLabel("Input Field 1-3")
        #right_layout.addWidget(input_field_label)

        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))
        for _ in range(3):
            lineEdit = QLineEdit()
            lineEdit.setPlaceholderText(f"Input Field {_}")
            lineEdit.setValidator(validator)
            right_layout.addWidget(lineEdit)

        # Add the left and right layouts to the container
        container_layout.addLayout(left_layout)
        container_layout.addLayout(right_layout)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        layout.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)


class AddLayerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Layer")

        layout = QVBoxLayout(self)

        #Dropdown menu for selecting options
        self.comboBox = QComboBox()

        #Options for combo box
        self.comboBox.addItems(["Holder 1", "Holder 2", "Holder 3"])
        layout.addWidget(self.comboBox)

        #Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        layout.addWidget(self.buttons)

        #Connect buttons
        self.buttons.accepted.connect(self.openConfigureAddLayerDialog)
        self.buttons.rejected.connect(self.reject)

        self.setMinimumSize(300, 100)

    def openConfigureAddLayerDialog(self):
        #open dialog when save is clicked, instead of accepting
        self.dialog = ConfigureAddLayerDialog(self)
        self.dialog.exec_()

class UsePresetDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Use Preset")

        layout = QVBoxLayout(self)

        #Dropdown menu for selecting options
        self.comboBox = QComboBox()

        #Options for it
        self.comboBox.addItems(["Standard Neural Network", "Convolutional NN", "Recurrent NN", "EEGNET-like"])
        layout.addWidget(self.comboBox)

        #Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        layout.addWidget(self.buttons)

        #Connect buttons
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.setMinimumSize(300, 100)


class ConfigureInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Input")

        layout = QVBoxLayout(self)

        # File selection components
        file_selection_layout = QHBoxLayout()
        self.file_path_label = QLabel("No directory selected")
        self.file_path_label.setWordWrap(True)  # Allow label to wrap if path is too long
        select_file_button = QPushButton("Select Directory")
        file_selection_layout.addWidget(select_file_button)
        file_selection_layout.addWidget(self.file_path_label, 1)  # The '1' makes the label expandable

        select_file_button.clicked.connect(self.selectDirectory)

        layout.addLayout(file_selection_layout)

        # Input Shape components grouped together
        input_shape_group = QGroupBox("Input Shape")
        input_shape_layout = QHBoxLayout(input_shape_group)

        width_input = QLineEdit()
        height_input = QLineEdit()

        validator = QRegularExpressionValidator(QRegularExpression("[0-9]{0,4}"))
        width_input.setValidator(validator)
        height_input.setValidator(validator)

        input_shape_layout.addWidget(width_input)
        input_shape_layout.addWidget(QLabel("X"))
        input_shape_layout.addWidget(height_input)

        layout.addWidget(input_shape_group)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def selectDirectory(self):

        file_path = QFileDialog.getExistingDirectory(self, "Select directory")
        if file_path:  # Only update the label if a file path was selected
            self.file_path_label.setText(file_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Main widget for the QMainWindow.
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        # Main layout for the central widget.
        main_layout = QVBoxLayout(main_widget)

        # Tab widget setup.
        tab_widget = QTabWidget()
        for i in range(1, 7):
            tab = QWidget()
            layout = QVBoxLayout(tab)
            label = QLabel(f"Content of Model {i}", tab)
            layout.addWidget(label)
            tab_widget.addTab(tab, f"Model {i}")

        # Adding the tab widget to the main layout with stretch factor.
        main_layout.addWidget(tab_widget, 1)  # Add stretch to make sure it expands

        # Buttons
        buttons_layout = QHBoxLayout()
        self.add_layer_button = QPushButton("Add Layer")
        self.remove_layer_button = QPushButton("Remove Layer")
        self.use_preset_button = QPushButton("Use Preset")
        self.configure_input_button = QPushButton("Configure Input")
        buttons = [self.add_layer_button, self.remove_layer_button, self.use_preset_button, self.configure_input_button]
        for button in buttons:
            if isinstance(button, QPushButton):
                buttons_layout.addWidget(button)

        # Add buttons layout below the tab widget.
        main_layout.addLayout(buttons_layout)

        # Connect the "Configure Input" button
        self.configure_input_button.clicked.connect(self.openConfigureInputModal)

        # Connect the "Use Preset" button
        self.use_preset_button.clicked.connect(self.openUsePresetModal)

        # Connect the "Add Layer" button
        self.add_layer_button.clicked.connect(self.openAddLayerModal)

        # Adjust the main window's size to ensure content is visible.
        self.setMinimumSize(800, 600)

    def openConfigureInputModal(self):
        dialog = ConfigureInputDialog(self)
        dialog.exec()


    def openUsePresetModal(self):
        dialog = UsePresetDialog(self)
        dialog.exec()

    def openAddLayerModal(self):
        dialog = AddLayerDialog(self)
        dialog.exec()

def parseData(rootPath):

    labelledData = []
    visited = set()

    for participant in os.listdir(rootPath):

        labels = os.path.join(rootPath, participant)
    
        for i, label in enumerate(os.listdir(labels)):
            if label not in visited:
                data = [[], label]
                labelledData.append(data)
                visited.add(label)

            files = os.path.join(labels, label)
            for file in os.listdir(files):
                updatedPath = os.path.join(files, file)

                labelledData[i][0].append(updatedPath)

    return labelledData

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("EEG App")
    window.show()
    sys.exit(app.exec())
