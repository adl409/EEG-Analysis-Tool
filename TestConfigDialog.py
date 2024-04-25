class TestConfigDialog(QDialog):

    def onValueChanged(self, value):
        # Update the label text to display the percentage
        self.label.setText(f'Data Testing Split: {value}%')
        #keeps selection to only ticks
        aligned_value = round(value / 5) * 5
        self.slider.setValue(aligned_value)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Testing")

        layout = QVBoxLayout(self)

        #Epoch testing
        self.epoch_group = QGroupBox("Epochs")
        self.epoch_group.setMaximumWidth(70)
        self.epoch_group.setMinimumWidth(50)
        self.epoch_layout = QHBoxLayout(self.epoch_group)
        self.epoch_input = QLineEdit()
        self.epoch_input.setMaximumWidth(50)
        self.epoch_input.setMinimumWidth(50)

        validator = QRegularExpressionValidator(QRegularExpression("[0-9]{0,4}"))
        self.epoch_input.setValidator(validator)

        self.epoch_layout.addWidget(self.epoch_input)
        layout.addWidget(self.epoch_group)



        #Shuffling
        self.checkbox = QCheckBox("Shuffle")
        layout.addWidget(self.checkbox)

        #slider for testing split
        self.label = QLabel('Data Testing Split: 20%')
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(20)
        self.slider.setMaximum(80)
        self.slider.setValue(20)
        self.slider.valueChanged.connect(self.onValueChanged)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(5)


        layout.addWidget(self.slider)

        #Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        layout.addWidget(self.buttons)

        #Connect buttons
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.setMinimumSize(300, 100)
