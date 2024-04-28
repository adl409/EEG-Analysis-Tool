import sys
import os
from PySide6.QtWidgets import QApplication, QSlider, QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QMainWindow, QTabWidget, QWidget, QSizePolicy, QGroupBox, QComboBox, QCheckBox
from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QRegularExpressionValidator

from layerClasses import *

# Dictionary that hold configuration values and will eventually be passed to the backend
models = {
    "input_parameters": {
        "input_shape1" : 0,
        "input_shape2" : 0,
        #EX dir
        "root_directory" : "",
        "normalized" : False
    },
    "test_parameters": {
        "test_split" : 0.20,
        "epochs" : 10,
        "shuffle" : True
    },
    "model_1" : {
        "save_model" : True, 
        "save_file" : "mod1",
        "active": True,
        "layers": [
            # Flatten(0), Dense(2, 64), Dense(1, 64, activation="relu")
        ]
    },
    "model_2" : {
        "save_model" : False,
        "save_file" : "mod2", 
        "active": False,
        "layers": [
            # Max_Pool_2d(1)
        ]
    },
    "model_3" : {
        "save_model" : False,
        "save_file" : "mod3", 
        "active": False,
        "layers": [
            # Dense(1, 64), Flatten(2)
        ]
    },
    "model_4" : {
        "save_model" : False,
        "save_file" : "mod4",
        "active": False,
        "layers": [
            # Dense(1, 64), Flatten(2)
        ]
    },
    "model_5" : {
        "save_model" : False,
        "save_file" : "mod5",
        "active": False,
        "layers": [
            # Dense(1, 64)
        ]
    }
}

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
        self.buttons.accepted.connect(self.saveConfig)
        self.buttons.rejected.connect(self.reject)

        self.setMinimumSize(300, 100)

    def saveConfig(self):

        models["test_parameters"]["test_split"] = self.slider.value() / 100
        models["test_parameters"]["epochs"] = float(self.epoch_input.text())
        models["test_parameters"]["shuffle"] = self.checkbox.isChecked()
        
        self.accept()

class ConfigureAddLayerDialog(QDialog):
    def __init__(self, layer_type, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Layer")

        self.layer_type = layer_type  # Store the selected layer type
        self.config = []              # Store the configuration for the Layer

        layout = QVBoxLayout(self)

        # Create and customize widgets based on layer type
        if self.layer_type == "Dense":
            self.configureDenseLayer(layout)
        elif self.layer_type == "Flatten":
            self.configureFlattenLayer(layout)
        elif self.layer_type == "Zero Padding 2d":
            self.configureZeroPadding2dLayer(layout)
        elif self.layer_type == "Average Pooling 2d":
            self.configureAveragePooling2dLayer(layout)
        elif self.layer_type == "Max Pooling 2d":
            self.configureMaxPooling2dLayer(layout)
        elif self.layer_type == "Convolution 2d":
            self.configureConvolution2dLayer(layout)
        elif self.layer_type == "Convolution 2d Transpose":
            self.configureConvolution2dTransposeLayer(layout)
        elif self.layer_type == "Depthwise Convolution 2d":
            self.configureDepthwiseConvolution2dLayer(layout)
        elif self.layer_type == "Separable Convolution 2d":
            self.configureSeparableConvolution2dLayer(layout)
        elif self.layer_type == "Convolution LSTM 2d":
            self.configureConvolutionLSTM2dLayer(layout)
        elif self.layer_type == "Simple RNN":
            self.configureSimpleRNNLayer(layout)
        elif self.layer_type == "LSTM":
            self.configureLSTMLayer(layout)
        elif self.layer_type == "GRU":
            self.configureGRULayer(layout)
        # Add conditions for other layer types

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        layout.addWidget(buttons)

        buttons.accepted.connect(self.saveLayer)    # Saving configuration function
        buttons.rejected.connect(self.reject)

    def configureDenseLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        units_label = QLabel("Units:")

        #add text box
        dense_units = QLineEdit()
        dense_units.setPlaceholderText("Units")
        dense_units.setValidator(validator)
        layout.addWidget(units_label)
        layout.addWidget(dense_units)

        self.config.append(dense_units)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        dense_activation = QComboBox()
        dense_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(dense_activation)

        self.config.append(dense_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform"])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform"])
        layout.addWidget(bias_initializer)
        
        self.config.append(bias_initializer)

        # Add widgets specific to configuring Dense layer
        pass

    def configureFlattenLayer(self, layout):

        flatten_label = QLabel("Flatten Layer (no extra parameters needed)")
        layout.addWidget(flatten_label)

        # Add widgets specific to configuring layer
        pass

    def configureZeroPadding2dLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))


        padding_label = QLabel("Padding: (x, y)")
        layout.addWidget(padding_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        zp2d_padding_x = QLineEdit()
        zp2d_padding_x.setValidator(validator)
        zp2d_padding_x.setPlaceholderText("Padding x value")
        layout.addWidget(zp2d_padding_x)

        self.config.append(zp2d_padding_x)

        zp2d_padding_y = QLineEdit()
        zp2d_padding_y.setValidator(validator)
        zp2d_padding_y.setPlaceholderText("Padding y value")
        layout.addWidget(zp2d_padding_y)

        self.config.append(zp2d_padding_y)

        #PROBABLY DOESN'T WORK RIGHT
        zp2d_padding = []
        zp2d_padding.append(zp2d_padding_x.text())
        zp2d_padding.append(zp2d_padding_y.text())

        #print(zp2d_padding)



        # Add widgets specific to configuring layer
        pass

    def configureAveragePooling2dLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))


        pool_size_label = QLabel("Pool Size: (x, y)")
        layout.addWidget(pool_size_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        ap2d_poolsize_x = QLineEdit()
        ap2d_poolsize_x.setValidator(validator)
        ap2d_poolsize_x.setPlaceholderText("Pool Size x value")
        layout.addWidget(ap2d_poolsize_x)

        self.config.append(ap2d_poolsize_x)

        ap2d_poolsize_y = QLineEdit()
        ap2d_poolsize_y.setValidator(validator)
        ap2d_poolsize_y.setPlaceholderText("Pool Size y value")
        layout.addWidget(ap2d_poolsize_y)

        self.config.append(ap2d_poolsize_y)

        #PROBABLY DOESN'T WORK RIGHT
        ap2d_poolsize = []
        ap2d_poolsize.append(ap2d_poolsize_x.text())
        ap2d_poolsize.append(ap2d_poolsize_y.text())


        strides_label = QLabel("Strides: (x, y)")
        layout.addWidget(strides_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        ap2d_strides_x = QLineEdit()
        ap2d_strides_x.setValidator(validator)
        ap2d_strides_x.setPlaceholderText("Strides x value")
        layout.addWidget(ap2d_strides_x)

        self.config.append(ap2d_strides_x)

        ap2d_strides_y = QLineEdit()
        ap2d_strides_y.setValidator(validator)
        ap2d_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(ap2d_strides_y)

        self.config.append(ap2d_strides_y)

        #PROBABLY DOESN'T WORK RIGHT
        ap2d_strides = []
        ap2d_strides.append(ap2d_strides_x.text())
        ap2d_strides.append(ap2d_strides_y.text())

        #padding dropdown
        padding_label = QLabel("Padding:")
        layout.addWidget(padding_label)
        ap2d_padding = QComboBox()
        ap2d_padding.addItems(["valid", "same"])
        layout.addWidget(ap2d_padding)

        self.config.append(ap2d_padding)


        # Add widgets specific to configuring layer
        pass

    def configureMaxPooling2dLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))


        pool_size_label = QLabel("Pool Size: (x, y)")
        layout.addWidget(pool_size_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        mp2d_poolsize_x = QLineEdit()
        mp2d_poolsize_x.setValidator(validator)
        mp2d_poolsize_x.setPlaceholderText("Pool Size x value")
        layout.addWidget(mp2d_poolsize_x)

        self.config.append(mp2d_poolsize_x)

        mp2d_poolsize_y = QLineEdit()
        mp2d_poolsize_y.setValidator(validator)
        mp2d_poolsize_y.setPlaceholderText("Pool Size y value")
        layout.addWidget(mp2d_poolsize_y)

        self.config.append(mp2d_poolsize_y)

        #PROBABLY DOESN'T WORK RIGHT
        mp2d_poolsize = []
        mp2d_poolsize.append(mp2d_poolsize_x.text())
        mp2d_poolsize.append(mp2d_poolsize_y.text())


        strides_label = QLabel("Strides: (x, y)")
        layout.addWidget(strides_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        mp2d_strides_x = QLineEdit()
        mp2d_strides_x.setValidator(validator)
        mp2d_strides_x.setPlaceholderText("Strides x value")
        layout.addWidget(mp2d_strides_x)

        self.config.append(mp2d_strides_x)

        mp2d_strides_y = QLineEdit()
        mp2d_strides_y.setValidator(validator)
        mp2d_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(mp2d_strides_y)

        self.config.append(mp2d_strides_y)

        #PROBABLY DOESN'T WORK RIGHT
        mp2d_strides = []
        mp2d_strides.append(mp2d_strides_x.text())
        mp2d_strides.append(mp2d_strides_y.text())

        #padding dropdown
        padding_label = QLabel("Padding:")
        layout.addWidget(padding_label)
        mp2d_padding = QComboBox()
        mp2d_padding.addItems(["valid", "same"])
        layout.addWidget(mp2d_padding)

        self.config.append(mp2d_padding)

        # Add widgets specific to configuring layer
        pass

    def configureConvolution2dLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        #filters
        filters_label = QLabel("Filters:")
        layout.addWidget(filters_label)

        #add text box
        c2d_filter = QLineEdit()
        c2d_filter.setPlaceholderText("Filters")
        c2d_filter.setValidator(validator)
        layout.addWidget(c2d_filter)

        self.config.append(c2d_filter)

        #kernel size
        kernelsize_label = QLabel("Kernel Size: (x, y)")
        layout.addWidget(kernelsize_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2d_kernelsize_x = QLineEdit()
        c2d_kernelsize_x.setValidator(validator)
        c2d_kernelsize_x.setPlaceholderText("Kernel Size x value")
        layout.addWidget(c2d_kernelsize_x)

        self.config.append(c2d_kernelsize_x)

        c2d_kernelsize_y = QLineEdit()
        c2d_kernelsize_y.setValidator(validator)
        c2d_kernelsize_y.setPlaceholderText("Kernel Size y value")
        layout.addWidget(c2d_kernelsize_y)

        self.config.append(c2d_kernelsize_y)

        #strides
        strides_label = QLabel("Strides: (x, y)")
        layout.addWidget(strides_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2d_strides_x = QLineEdit()
        c2d_strides_x.setValidator(validator)
        c2d_strides_x.setPlaceholderText("Strides x value")
        layout.addWidget(c2d_strides_x)

        self.config.append(c2d_strides_x)

        c2d_strides_y = QLineEdit()
        c2d_strides_y.setValidator(validator)
        c2d_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(c2d_strides_y)

        self.config.append(c2d_strides_y)

        #PROBABLY DOESN'T WORK RIGHT
        c2d_strides = []
        c2d_strides.append(c2d_strides_x.text())
        c2d_strides.append(c2d_strides_y.text())

        #padding dropdown
        padding_label = QLabel("Padding:")
        layout.addWidget(padding_label)
        c2d_padding = QComboBox()
        c2d_padding.addItems(["valid", "same"])
        layout.addWidget(c2d_padding)

        self.config.append(c2d_padding)

        #dialation rate
        dialationrate_label = QLabel("Dialation Rate: (x, y)")
        layout.addWidget(dialationrate_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2d_dialationrate_x = QLineEdit()
        c2d_dialationrate_x.setValidator(validator)
        c2d_dialationrate_x.setPlaceholderText("Dialation Rate x value")
        layout.addWidget(c2d_dialationrate_x)

        self.config.append(c2d_dialationrate_x)

        c2d_dialationrate_y = QLineEdit()
        c2d_dialationrate_y.setValidator(validator)
        c2d_dialationrate_y.setPlaceholderText("Dialation Rate y value")
        layout.addWidget(c2d_dialationrate_y)

        self.config.append(c2d_dialationrate_y)

        #groups
        groups_label = QLabel("Groups:")
        layout.addWidget(groups_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY
        c2d_groups = QLineEdit()
        c2d_groups.setPlaceholderText("Groups")
        c2d_groups.setValidator(validator)
        layout.addWidget(c2d_groups)

        self.config.append(c2d_groups)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        c2d_activation = QComboBox()
        c2d_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(c2d_activation)

        self.config.append(c2d_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        # Add widgets specific to configuring layer
        pass

    def configureConvolution2dTransposeLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        #filters
        filters_label = QLabel("Filters:")
        layout.addWidget(filters_label)

        #add text box
        c2dt_filter = QLineEdit()
        c2dt_filter.setPlaceholderText("Filters")
        c2dt_filter.setValidator(validator)
        layout.addWidget(c2dt_filter)

        self.config.append(c2dt_filter)

        #kernel size
        kernelsize_label = QLabel("Kernel Size: (x, y)")
        layout.addWidget(kernelsize_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2dt_kernelsize_x = QLineEdit()
        c2dt_kernelsize_x.setValidator(validator)
        c2dt_kernelsize_x.setPlaceholderText("Kernel Size x value")
        layout.addWidget(c2dt_kernelsize_x)

        self.config.append(c2dt_kernelsize_x)

        c2dt_kernelsize_y = QLineEdit()
        c2dt_kernelsize_y.setValidator(validator)
        c2dt_kernelsize_y.setPlaceholderText("Kernel Size y value")
        layout.addWidget(c2dt_kernelsize_y)

        self.config.append(c2dt_kernelsize_y)

        #strides
        strides_label = QLabel("Strides: (x, y)")
        layout.addWidget(strides_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2dt_strides_x = QLineEdit()
        c2dt_strides_x.setValidator(validator)
        c2dt_strides_x.setPlaceholderText("Strides x value")
        layout.addWidget(c2dt_strides_x)

        self.config.append(c2dt_strides_x)

        c2dt_strides_y = QLineEdit()
        c2dt_strides_y.setValidator(validator)
        c2dt_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(c2dt_strides_y)

        self.config.append(c2dt_strides_y)

        #PROBABLY DOESN'T WORK RIGHT
        c2dt_strides = []
        c2dt_strides.append(c2dt_strides_x.text())
        c2dt_strides.append(c2dt_strides_y.text())

        #padding dropdown
        padding_label = QLabel("Padding:")
        layout.addWidget(padding_label)
        c2dt_padding = QComboBox()
        c2dt_padding.addItems(["valid", "same"])
        layout.addWidget(c2dt_padding)

        self.config.append(c2dt_padding)

        #dialation rate
        dialationrate_label = QLabel("Dialation Rate: (x, y)")
        layout.addWidget(dialationrate_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2dt_dialationrate_x = QLineEdit()
        c2dt_dialationrate_x.setValidator(validator)
        c2dt_dialationrate_x.setPlaceholderText("Dialation Rate x value")
        layout.addWidget(c2dt_dialationrate_x)

        self.config.append(c2dt_dialationrate_x)

        c2dt_dialationrate_y = QLineEdit()
        c2dt_dialationrate_y.setValidator(validator)
        c2dt_dialationrate_y.setPlaceholderText("Dialation Rate y value")
        layout.addWidget(c2dt_dialationrate_y)

        self.config.append(c2dt_dialationrate_y)

        #groups
        groups_label = QLabel("Groups:")
        layout.addWidget(groups_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY
        c2dt_groups = QLineEdit()
        c2dt_groups.setPlaceholderText("Groups")
        c2dt_groups.setValidator(validator)
        layout.addWidget(c2dt_groups)

        self.config.append(c2dt_groups)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        c2dt_activation = QComboBox()
        c2dt_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(c2dt_activation)

        self.config.append(c2dt_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)



        # Add widgets specific to configuring layer
        pass

    def configureDepthwiseConvolution2dLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        #filters
        filters_label = QLabel("Filters:")
        layout.addWidget(filters_label)

        #add text box
        dc2d_filter = QLineEdit()
        dc2d_filter.setPlaceholderText("Filters")
        dc2d_filter.setValidator(validator)
        layout.addWidget(dc2d_filter)

        self.config.append(dc2d_filter)

        #kernel size
        kernelsize_label = QLabel("Kernel Size: (x, y)")
        layout.addWidget(kernelsize_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        dc2d_kernelsize_x = QLineEdit()
        dc2d_kernelsize_x.setValidator(validator)
        dc2d_kernelsize_x.setPlaceholderText("Kernel Size x value")
        layout.addWidget(dc2d_kernelsize_x)

        self.config.append(dc2d_kernelsize_x)

        dc2d_kernelsize_y = QLineEdit()
        dc2d_kernelsize_y.setValidator(validator)
        dc2d_kernelsize_y.setPlaceholderText("Kernel Size y value")
        layout.addWidget(dc2d_kernelsize_y)

        self.config.append(dc2d_kernelsize_y)

        #strides
        strides_label = QLabel("Strides: (x, y)")
        layout.addWidget(strides_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        dc2d_strides_x = QLineEdit()
        dc2d_strides_x.setValidator(validator)
        dc2d_strides_x.setPlaceholderText("Strides x value")
        layout.addWidget(dc2d_strides_x)

        self.config.append(dc2d_strides_x)

        dc2d_strides_y = QLineEdit()
        dc2d_strides_y.setValidator(validator)
        dc2d_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(dc2d_strides_y)

        self.config.append(dc2d_strides_y)

        #PROBABLY DOESN'T WORK RIGHT
        dc2d_strides = []
        dc2d_strides.append(dc2d_strides_x.text())
        dc2d_strides.append(dc2d_strides_y.text())

        #padding dropdown
        padding_label = QLabel("Padding:")
        layout.addWidget(padding_label)
        dc2d_padding = QComboBox()
        dc2d_padding.addItems(["valid", "same"])
        layout.addWidget(dc2d_padding)

        self.config.append(dc2d_padding)

        #dialation rate
        dialationrate_label = QLabel("Dialation Rate: (x, y)")
        layout.addWidget(dialationrate_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        dc2d_dialationrate_x = QLineEdit()
        dc2d_dialationrate_x.setValidator(validator)
        dc2d_dialationrate_x.setPlaceholderText("Dialation Rate x value")
        layout.addWidget(dc2d_dialationrate_x)

        self.config.append(dc2d_dialationrate_x)

        dc2d_dialationrate_y = QLineEdit()
        dc2d_dialationrate_y.setValidator(validator)
        dc2d_dialationrate_y.setPlaceholderText("Dialation Rate y value")
        layout.addWidget(dc2d_dialationrate_y)

        self.config.append(dc2d_dialationrate_y)

        #depth multiplier
        depthmultiplier_label = QLabel("Depth Multiplier:")
        layout.addWidget(depthmultiplier_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY
        dc2d_depthmultiplier = QLineEdit()
        dc2d_depthmultiplier.setPlaceholderText("Depth Multiplier")
        dc2d_depthmultiplier.setValidator(validator)
        layout.addWidget(dc2d_depthmultiplier)

        self.config.append(dc2d_depthmultiplier)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        dc2d_activation = QComboBox()
        dc2d_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(dc2d_activation)

        self.config.append(dc2d_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #depthwise initializer
        d_initializer_label = QLabel("Depthwise Initializer:")
        layout.addWidget(d_initializer_label)
        depthwise_initializer = QComboBox()
        depthwise_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(depthwise_initializer)

        self.config.append(depthwise_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        # Add widgets specific to configuring layer
        pass

    def configureSeparableConvolution2dLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        #filters
        filters_label = QLabel("Filters:")
        layout.addWidget(filters_label)

        #add text box
        sc2d_filter = QLineEdit()
        sc2d_filter.setPlaceholderText("Filters")
        sc2d_filter.setValidator(validator)
        layout.addWidget(sc2d_filter)

        self.config.append(sc2d_filter)

        #kernel size
        kernelsize_label = QLabel("Kernel Size: (x, y)")
        layout.addWidget(kernelsize_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        sc2d_kernelsize_x = QLineEdit()
        sc2d_kernelsize_x.setValidator(validator)
        sc2d_kernelsize_x.setPlaceholderText("Kernel Size x value")
        layout.addWidget(sc2d_kernelsize_x)

        self.config.append(sc2d_kernelsize_x)

        sc2d_kernelsize_y = QLineEdit()
        sc2d_kernelsize_y.setValidator(validator)
        sc2d_kernelsize_y.setPlaceholderText("Kernel Size y value")
        layout.addWidget(sc2d_kernelsize_y)

        self.config.append(sc2d_kernelsize_y)

        #strides
        strides_label = QLabel("Strides: (x, y)")
        layout.addWidget(strides_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        sc2d_strides_x = QLineEdit()
        sc2d_strides_x.setValidator(validator)
        sc2d_strides_x.setPlaceholderText("Strides x value")
        layout.addWidget(sc2d_strides_x)

        self.config.append(sc2d_strides_x)

        sc2d_strides_y = QLineEdit()
        sc2d_strides_y.setValidator(validator)
        sc2d_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(sc2d_strides_y)

        self.config.append(sc2d_strides_y)

        #PROBABLY DOESN'T WORK RIGHT
        sc2d_strides = []
        sc2d_strides.append(sc2d_strides_x.text())
        sc2d_strides.append(sc2d_strides_y.text())

        #padding dropdown
        padding_label = QLabel("Padding:")
        layout.addWidget(padding_label)
        sc2d_padding = QComboBox()
        sc2d_padding.addItems(["valid", "same"])
        layout.addWidget(sc2d_padding)

        self.config.append(sc2d_padding)

        #dialation rate
        dialationrate_label = QLabel("Dialation Rate: (x, y)")
        layout.addWidget(dialationrate_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        sc2d_dialationrate_x = QLineEdit()
        sc2d_dialationrate_x.setValidator(validator)
        sc2d_dialationrate_x.setPlaceholderText("Dialation Rate x value")
        layout.addWidget(sc2d_dialationrate_x)

        self.config.append(sc2d_dialationrate_x)

        sc2d_dialationrate_y = QLineEdit()
        sc2d_dialationrate_y.setValidator(validator)
        sc2d_dialationrate_y.setPlaceholderText("Dialation Rate y value")
        layout.addWidget(sc2d_dialationrate_y)

        self.config.append(sc2d_dialationrate_y)

        #depth multiplier
        depthmultiplier_label = QLabel("Depth Multiplier:")
        layout.addWidget(depthmultiplier_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY
        sc2d_depthmultiplier = QLineEdit()
        sc2d_depthmultiplier.setPlaceholderText("Depth Multiplier")
        sc2d_depthmultiplier.setValidator(validator)
        layout.addWidget(sc2d_depthmultiplier)

        self.config.append(sc2d_depthmultiplier)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        sc2d_activation = QComboBox()
        sc2d_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(sc2d_activation)

        self.config.append(sc2d_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #depthwise initializer
        d_initializer_label = QLabel("Depthwise Initializer:")
        layout.addWidget(d_initializer_label)
        depthwise_initializer = QComboBox()
        depthwise_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(depthwise_initializer)

        self.config.append(depthwise_initializer)

        #pointwise initializer
        p_initializer_label = QLabel("Pointwise Initializer:")
        layout.addWidget(p_initializer_label)
        pointwise_initializer = QComboBox()
        pointwise_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(pointwise_initializer)

        self.config.append(pointwise_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        # Add widgets specific to configuring layer
        pass

    def configureConvolutionLSTM2dLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        float_validator = QRegularExpressionValidator(QRegularExpression('^(\d)*(\.)?([0-9]{1})?$'))

        #filters
        filters_label = QLabel("Filters:")
        layout.addWidget(filters_label)

        #add text box
        clstm2d_filter = QLineEdit()
        clstm2d_filter.setPlaceholderText("Filters")
        clstm2d_filter.setValidator(validator)
        layout.addWidget(clstm2d_filter)

        self.config.append(clstm2d_filter)

        #kernel size
        kernelsize_label = QLabel("Kernel Size: (x, y)")
        layout.addWidget(kernelsize_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        clstm2d_kernelsize_x = QLineEdit()
        clstm2d_kernelsize_x.setValidator(validator)
        clstm2d_kernelsize_x.setPlaceholderText("Kernel Size x value")
        layout.addWidget(clstm2d_kernelsize_x)

        self.config.append(clstm2d_kernelsize_x)

        clstm2d_kernelsize_y = QLineEdit()
        clstm2d_kernelsize_y.setValidator(validator)
        clstm2d_kernelsize_y.setPlaceholderText("Kernel Size y value")
        layout.addWidget(clstm2d_kernelsize_y)

        self.config.append(clstm2d_kernelsize_y)

        #strides
        strides_label = QLabel("Strides: (x, y)")
        layout.addWidget(strides_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        clstm2d_strides_x = QLineEdit()
        clstm2d_strides_x.setValidator(validator)
        clstm2d_strides_x.setPlaceholderText("Strides x value")
        layout.addWidget(clstm2d_strides_x)

        self.config.append(clstm2d_strides_x)

        clstm2d_strides_y = QLineEdit()
        clstm2d_strides_y.setValidator(validator)
        clstm2d_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(clstm2d_strides_y)

        self.config.append(clstm2d_strides_y)

        #PROBABLY DOESN'T WORK RIGHT
        clstm2d_strides = []
        clstm2d_strides.append(clstm2d_strides_x.text())
        clstm2d_strides.append(clstm2d_strides_y.text())

        #padding dropdown
        padding_label = QLabel("Padding:")
        layout.addWidget(padding_label)
        clstm2d_padding = QComboBox()
        clstm2d_padding.addItems(["valid", "same"])
        layout.addWidget(clstm2d_padding)

        self.config.append(clstm2d_padding)

        #dialation rate
        dialationrate_label = QLabel("Dialation Rate: (x, y)")
        layout.addWidget(dialationrate_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        clstm2d_dialationrate_x = QLineEdit()
        clstm2d_dialationrate_x.setValidator(validator)
        clstm2d_dialationrate_x.setPlaceholderText("Dialation Rate x value")
        layout.addWidget(clstm2d_dialationrate_x)

        self.config.append(clstm2d_dialationrate_x)

        clstm2d_dialationrate_y = QLineEdit()
        clstm2d_dialationrate_y.setValidator(validator)
        clstm2d_dialationrate_y.setPlaceholderText("Dialation Rate y value")
        layout.addWidget(clstm2d_dialationrate_y)

        self.config.append(clstm2d_dialationrate_y)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        clstm2d_activation = QComboBox()
        clstm2d_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(clstm2d_activation)

        self.config.append(clstm2d_activation)

        #recurrent activation dropdown
        recurrent_activation_label = QLabel("Recurrent Activation Type:")
        layout.addWidget(recurrent_activation_label)
        clstm2d_recurrent_activation = QComboBox()
        clstm2d_recurrent_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(clstm2d_recurrent_activation)

        self.config.append(clstm2d_recurrent_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("Recurrent Initializer:")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(recurrent_initializer)

        self.config.append(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        #dropout
        dropout_label = QLabel("Dropout:")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        clstm2d_dropout = QLineEdit()
        clstm2d_dropout.setPlaceholderText("Dropout")
        # clstm2d_dropout.setValidator(float_validator)
        layout.addWidget(clstm2d_dropout)

        self.config.append(clstm2d_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("Recurrent Dropout:")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        clstm2d_recurrent_dropout = QLineEdit()
        clstm2d_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        # clstm2d_recurrent_dropout.setValidator(float_validator)
        layout.addWidget(clstm2d_recurrent_dropout)

        self.config.append(clstm2d_recurrent_dropout)

        #seed
        seed_label = QLabel("Seed:")
        layout.addWidget(seed_label)

        #add text box
        clstm2d_seed = QLineEdit()
        clstm2d_seed.setPlaceholderText("Seed")
        clstm2d_seed.setValidator(validator)
        layout.addWidget(clstm2d_seed)

        self.config.append(clstm2d_seed)

        # Add widgets specific to configuring layer
        pass

    def configureSimpleRNNLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        float_validator = QRegularExpressionValidator(QRegularExpression('^(\d)*(\.)?([0-9]{1})?$'))

        #units
        units_label = QLabel("Units:")
        layout.addWidget(units_label)

        #add text box
        srnn_units = QLineEdit()
        srnn_units.setPlaceholderText("Units")
        srnn_units.setValidator(validator)
        layout.addWidget(srnn_units)

        self.config.append(srnn_units)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        srnn_activation = QComboBox()
        srnn_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(srnn_activation)

        self.config.append(srnn_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("Recurrent Initializer:")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(recurrent_initializer)

        self.config.append(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        #dropout
        dropout_label = QLabel("Dropout:")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        srnn_dropout = QLineEdit()
        srnn_dropout.setPlaceholderText("Dropout")
        # srnn_dropout.setValidator(float_validator)
        layout.addWidget(srnn_dropout)

        self.config.append(srnn_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("Recurrent Dropout:")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        srnn_recurrent_dropout = QLineEdit()
        srnn_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        # srnn_recurrent_dropout.setValidator(float_validator)
        layout.addWidget(srnn_recurrent_dropout)

        self.config.append(srnn_recurrent_dropout)

        #seed
        seed_label = QLabel("Seed:")
        layout.addWidget(seed_label)

        #add text box
        srnn_seed = QLineEdit()
        srnn_seed.setPlaceholderText("Seed")
        srnn_seed.setValidator(validator)
        layout.addWidget(srnn_seed)

        self.config.append(srnn_seed)

        # Add widgets specific to configuring layer
        pass

    def configureLSTMLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        float_validator = QRegularExpressionValidator(QRegularExpression('^(\d)*(\.)?([0-9]{1})?$'))

        #units
        units_label = QLabel("Units:")
        layout.addWidget(units_label)

        #add text box
        lstm_units = QLineEdit()
        lstm_units.setPlaceholderText("Units")
        lstm_units.setValidator(validator)
        layout.addWidget(lstm_units)

        self.config.append(lstm_units)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        lstm_activation = QComboBox()
        lstm_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(lstm_activation)

        self.config.append(lstm_activation)

        #recurrent activation dropdown
        recurrent_activation_label = QLabel("Recurrent Activation Type:")
        layout.addWidget(recurrent_activation_label)
        lstm_recurrent_activation = QComboBox()
        lstm_recurrent_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(lstm_recurrent_activation)

        self.config.append(lstm_recurrent_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("Recurrent Initializer:")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(recurrent_initializer)

        self.config.append(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        #unit forget bias dropdown
        unit_forget_bias_label = QLabel("Use Bias:")
        layout.addWidget(unit_forget_bias_label)
        unit_forget_bias = QComboBox()
        unit_forget_bias.addItems(["True", "False"])
        layout.addWidget(unit_forget_bias)

        self.config.append(unit_forget_bias)


        #dropout
        dropout_label = QLabel("Dropout:")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        lstm_dropout = QLineEdit()
        lstm_dropout.setPlaceholderText("Dropout")
        # lstm_dropout.setValidator(float_validator)
        layout.addWidget(lstm_dropout)

        self.config.append(lstm_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("Recurrent Dropout:")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        lstm_recurrent_dropout = QLineEdit()
        lstm_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        # lstm_recurrent_dropout.setValidator(float_validator)
        layout.addWidget(lstm_recurrent_dropout)

        self.config.append(lstm_recurrent_dropout)

        #seed
        seed_label = QLabel("Seed:")
        layout.addWidget(seed_label)

        #add text box
        lstm_seed = QLineEdit()
        lstm_seed.setPlaceholderText("Seed")
        lstm_seed.setValidator(validator)
        layout.addWidget(lstm_seed)

        self.config.append(lstm_seed)

        # Add widgets specific to configuring layer
        pass

    def configureGRULayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        float_validator = QRegularExpressionValidator(QRegularExpression('^(\d)*(\.)?([0-9]{1})?$'))

        #units
        units_label = QLabel("Units:")
        layout.addWidget(units_label)

        #add text box
        gru_units = QLineEdit()
        gru_units.setPlaceholderText("Units")
        gru_units.setValidator(validator)
        layout.addWidget(gru_units)

        self.config.append(gru_units)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        gru_activation = QComboBox()
        gru_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(gru_activation)

        self.config.append(gru_activation)

        #recurrent activation dropdown
        recurrent_activation_label = QLabel("Recurrent Activation Type:")
        layout.addWidget(recurrent_activation_label)
        gru_recurrent_activation = QComboBox()
        gru_recurrent_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(gru_recurrent_activation)

        self.config.append(gru_recurrent_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("Recurrent Initializer:")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(recurrent_initializer)

        self.config.append(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        gru_unit_forget_bias = QCheckBox("Unit Forget Bias")
        layout.addWidget(gru_unit_forget_bias)

        self.config.append(gru_unit_forget_bias)

        #dropout
        dropout_label = QLabel("Dropout:")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        gru_dropout = QLineEdit()
        gru_dropout.setPlaceholderText("Dropout")
        # gru_dropout.setValidator(float_validator)
        layout.addWidget(gru_dropout)

        self.config.append(gru_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("Recurrent Dropout:")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        gru_recurrent_dropout = QLineEdit()
        gru_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        # lstm_recurrent_dropout.setValidator(float_validator)
        layout.addWidget(gru_recurrent_dropout)

        self.config.append(gru_recurrent_dropout)

        #seed
        seed_label = QLabel("Seed:")
        layout.addWidget(seed_label)

        #add text box
        gru_seed = QLineEdit()
        gru_seed.setPlaceholderText("Seed")
        gru_seed.setValidator(validator)
        layout.addWidget(gru_seed)

        self.config.append(gru_seed)

        #reset after dropdown
        reset_after_label = QLabel("Reset After:")
        layout.addWidget(reset_after_label)
        reset_after = QComboBox()
        reset_after.addItems(["True", "False"])
        layout.addWidget(reset_after)

        self.config.append(reset_after)

        # Add widgets specific to configuring layer
        pass

    def saveLayer(self):
        # Append to main dictionary the layer type the user wants to add
        length = len(models["model_1"]["layers"])
        if self.layer_type == "Dense":
            models["model_1"]["layers"].append(
                Dense(
                    length, 
                    int(float(self.config[0].text())), 
                    self.config[1].currentText(), 
                    True if self.config[2].currentText() == "True" else False, 
                    self.config[3].currentText(),
                    self.config[4].currentText()
                    )
                )
        elif self.layer_type == "Flatten":
            models["model_1"]["layers"].append(
                Flatten(
                    length
                    )
                )
        elif self.layer_type == "Zero Padding 2d":
            models["model_1"]["layers"].append(
                Zero_Padding_2d(
                    length, 
                    (int(float(self.config[0].text())), int(float(self.config[1].text())))
                    )
                )
        elif self.layer_type == "Average Pooling 2d":
            models["model_1"]["layers"].append(
                Average_Pooling_2d(
                    length, 
                    (int(float(self.config[0].text())), int(float(self.config[1].text()))), 
                    (int(float(self.config[2].text())), int(float(self.config[3].text()))), 
                    self.config[4].currentText()
                    )
                )
        elif self.layer_type == "Max Pooling 2d":
            models["model_1"]["layers"].append(
                Max_Pool_2d(
                    length, 
                    (int(float(self.config[0].text())), int(float(self.config[1].text()))), 
                    (int(float(self.config[2].text())), int(float(self.config[3].text()))), 
                    self.config[4].currentText()
                    )
                )
        elif self.layer_type == "Convolution 2d":
            models["model_1"]["layers"].append(
                Convolution_2d(
                    length, 
                    int(float(self.config[0].text())), 
                    (int(float(self.config[1].text())), int(float(self.config[2].text()))), 
                    (int(float(self.config[3].text())), int(float(self.config[4].text()))), 
                    self.config[5].currentText(), 
                    (int(float(self.config[6].text())), int(float(self.config[7].text()))), 
                    int(float(self.config[8].text())), 
                    self.config[9].currentText(), 
                    self.config[10].currentText(), 
                    self.config[11].currentText(), 
                    self.config[12].currentText()
                    )
                )
        elif self.layer_type == "Convolution 2d Transpose":
            models["model_1"]["layers"].append(
                Convolution_2d_Transpose(
                    length, 
                    int(float(self.config[0].text())), 
                    (int(float(self.config[1].text())), int(float(self.config[2].text()))), 
                    (int(float(self.config[3].text())), int(float(self.config[4].text()))), 
                    self.config[5].currentText(), 
                    (int(float(self.config[6].text())), int(float(self.config[7].text()))), 
                    int(float(self.config[8].text())), 
                    self.config[9].currentText(), 
                    self.config[10].currentText(), 
                    self.config[11].currentText(), 
                    self.config[12].currentText()
                    )
                )
        elif self.layer_type == "Depthwise Convolution 2d":
            models["model_1"]["layers"].append(
                Depthwise_Conv_2d(
                    length, 
                    int(float(self.config[0].text())), 
                    (int(float(self.config[1].text())), int(float(self.config[2].text()))), 
                    (int(float(self.config[3].text())), int(float(self.config[4].text()))), 
                    self.config[5].currentText(), 
                    (int(float(self.config[6].text())), int(float(self.config[7].text()))), 
                    int(float(self.config[8].text())), 
                    self.config[9].currentText(), 
                    self.config[10].currentText(), 
                    self.config[11].currentText(), 
                    self.config[12].currentText()
                    )
                )
        elif self.layer_type == "Separable Convolution 2d":
            models["model_1"]["layers"].append(
                Separable_Conv_2d(
                    length, 
                    int(float(self.config[0].text())), 
                    (int(float(self.config[1].text())), int(float(self.config[2].text()))), 
                    (int(float(self.config[3].text())), int(float(self.config[4].text()))), 
                    self.config[5].currentText(), 
                    (int(float(self.config[6].text())), int(float(self.config[7].text()))), 
                    int(float(self.config[8].text())), 
                    self.config[9].currentText(), 
                    self.config[10].currentText(), 
                    self.config[11].currentText(), 
                    self.config[12].currentText(), 
                    self.config[13].currentText()
                    )
                )
        elif self.layer_type == "Convolution LSTM 2d":
            models["model_1"]["layers"].append(
                Conv_LSTM_2d(
                    length, 
                    int(float(self.config[0].text())), 
                    (int(float(self.config[1].text())), int(float(self.config[2].text()))), 
                    (int(float(self.config[3].text())), int(float(self.config[4].text()))),  
                    self.config[5].currentText(), 
                    (int(float(self.config[6].text())), int(float(self.config[7].text()))), 
                    self.config[8].currentText(), 
                    self.config[9].currentText(), 
                    self.config[10].currentText(), 
                    self.config[11].currentText(), 
                    self.config[12].currentText(), 
                    self.config[13].currentText(), 
                    float(self.config[14].text()), 
                    float(self.config[15].text()), 
                    int(float(self.config[16].text()))
                    )
                )
        elif self.layer_type == "Simple RNN":
            models["model_1"]["layers"].append(
                SimpleRNN(
                    length, 
                    int(float(self.config[0].text())), 
                    self.config[1].currentText(), 
                    self.config[2].currentText(), 
                    self.config[3].currentText(), 
                    self.config[4].currentText(), 
                    self.config[5].currentText(), 
                    float(self.config[6].text()), 
                    float(self.config[7].text()), 
                    int(float(self.config[8].text()))
                    )
                )
        elif self.layer_type == "LSTM":
            models["model_1"]["layers"].append(
                LSTM(
                    length, 
                    int(float(self.config[0].text())), 
                    self.config[1].currentText(), 
                    self.config[2].currentText(), 
                    self.config[3].currentText(), 
                    self.config[4].currentText(), 
                    self.config[5].currentText(), 
                    self.config[6].currentText(), 
                    self.config[7].currentText(), 
                    float(self.config[8].text()), 
                    float(self.config[9].text()), 
                    int(float(self.config[10].text()))
                    )
                )
        elif self.layer_type == "GRU":
            models["model_1"]["layers"].append(
                GRU(
                    length, 
                    int(float(self.config[0].text())), 
                    self.config[1].currentText(), 
                    self.config[2].currentText(), 
                    self.config[3].currentText(), 
                    self.config[4].currentText(), 
                    self.config[5].currentText(), 
                    self.config[6].currentText(), 
                    self.config[7].isChecked(), 
                    float(self.config[8].text()), 
                    float(self.config[9].text()), 
                    int(float(self.config[10].text())), 
                    self.config[11].currentText()
                    )
                )

        self.accept()

class AddLayerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Layer")

        layout = QVBoxLayout(self)

        self.comboBox = QComboBox()
        self.comboBox.addItems(["Dense", "Flatten", "Zero Padding 2d", "Average Pooling 2d", "Max Pooling 2d", "Convolution 2d", "Convolution 2d Transpose", "Depthwise Convolution 2d", "Separable Convolution 2d", "Convolution LSTM 2d", "Simple RNN", "LSTM", "GRU"])
        layout.addWidget(self.comboBox)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.openConfigureAddLayerDialog)
        self.buttons.rejected.connect(self.reject)

        self.comboBox.currentTextChanged.connect(self.updateConfigureDialog)

        self.setMinimumSize(300, 100)

    def openConfigureAddLayerDialog(self):
        dialog = ConfigureAddLayerDialog(self.comboBox.currentText(), self)
        dialog.exec()

    def updateConfigureDialog(self, layer_type):
        # Update the configuration dialog when the layer type is changed
        pass

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

        self.config = []

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

        self.config.append(width_input)
        self.config.append(height_input)

        input_shape_layout.addWidget(width_input)
        input_shape_layout.addWidget(QLabel("X"))
        input_shape_layout.addWidget(height_input)

        layout.addWidget(input_shape_group)

        normalize = QCheckBox("Normalize")
        layout.addWidget(normalize)

        self.config.append(normalize)

            # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(self.saveConfig) # Calls the saveConfig function once save button is pressed
        buttons.rejected.connect(self.reject)
        
    def selectDirectory(self):
        # Prompts a select directory screen
        file_path = QFileDialog.getExistingDirectory(self, "Select directory")
        if file_path:  # Only update the label if a file path was selected
            self.file_path_label.setText(file_path)
            models["input_parameters"]["root_directory"] = file_path # Saves file path to main dictionary

    def saveConfig(self):
        # Saves the Input shape size and normalization to main dictionary.
        models["input_parameters"]["input_shape1"] = self.config[0].text() if self.config[0].text() != "" else models["input_parameters"]["input_shape1"]
        models["input_parameters"]["input_shape2"] = self.config[1].text() if self.config[1].text() != "" else models["input_parameters"]["input_shape2"]
        models["input_parameters"]["normalized"] = self.config[2].isChecked()

        self.accept()

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
        self.test_input_button = QPushButton("Configure Testing")
        buttons = [self.add_layer_button, self.remove_layer_button, self.use_preset_button, self.configure_input_button, self.test_input_button]
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

        self.test_input_button.clicked.connect(self.openTestConfigModal)

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
    
    def openTestConfigModal(self):
        dialog = TestConfigDialog(self)
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

def getConfigs():
    return models

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("EEG App")
    window.show()
    sys.exit(app.exec())
