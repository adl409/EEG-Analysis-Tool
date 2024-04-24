import sys
import os
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QMainWindow, QTabWidget, QWidget, QSizePolicy, QGroupBox, QComboBox, QTableView, QFrame
from PySide6.QtCore import QRegularExpression, Signal, QObject, QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QRegularExpressionValidator

class LayerTableModel(QAbstractTableModel):
    def __init__(self, data = []):
        super().__init__()
        self._data = data
        self.headers = ["Layer Name", "Units", "Activation", "Use Bias", "Kernel Initializer", "Bias Initializer"]

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]
        return None

    def rowCount(self, index=QModelIndex()):
        return len(self._data)

    def columnCount(self, index=QModelIndex()):
        return len(self.headers) if self._data else 0

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return None

    def addLayer(self, layer_data):

        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self._data.append([layer_data.get(header, '') for header in self.headers])
        self.endInsertRows()

class ConfigureAddLayerDialog(QDialog):
    data_collected = Signal(dict) #custom signal emitting a dictionary

    def __init__(self, layer_type, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Layer")

        self.layer_type = layer_type  # Store the selected layer type

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

        buttons.accepted.connect(self.collectData)
        #buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)


    #CHANGE WITH NATHAN'S DICTIONARY EVENTUALLY
    def collectData(self):
        #NOTE: hard coded dense for now, go through and add layer name to each function after this one.
        layer_data = {
            "Layer Name": "Dense",
            "Units": self.dense_units.text(),
            "Activation": self.dense_activation.currentText(),
            "Use Bias": self.use_bias.currentText(),
            "Kernel Initializer": self.kernel_initializer.currentText(),
            "Bias Initializer": self.bias_initializer.currentText()
        }
        self.data_collected.emit(layer_data)
        self.accept()

    def configureDenseLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        units_label = QLabel("Units:")

        #add text box
        self.dense_units = QLineEdit()
        self.dense_units.setPlaceholderText("Units")
        self.dense_units.setValidator(validator)
        layout.addWidget(units_label)
        layout.addWidget(self.dense_units)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        self.dense_activation = QComboBox()
        self.dense_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(self.dense_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        self.use_bias = QComboBox()
        self.use_bias.addItems(["True", "False"])
        layout.addWidget(self.use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        self.kernel_initializer = QComboBox()
        self.kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform"])
        layout.addWidget(self.kernel_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        self.bias_initializer = QComboBox()
        self.bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform"])
        layout.addWidget(self.bias_initializer)

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

        zp2d_padding_y = QLineEdit()
        zp2d_padding_y.setValidator(validator)
        zp2d_padding_y.setPlaceholderText("Padding y value")
        layout.addWidget(zp2d_padding_y)

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

        ap2d_poolsize_y = QLineEdit()
        ap2d_poolsize_y.setValidator(validator)
        ap2d_poolsize_y.setPlaceholderText("Pool Size y value")
        layout.addWidget(ap2d_poolsize_y)

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

        ap2d_strides_y = QLineEdit()
        ap2d_strides_y.setValidator(validator)
        ap2d_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(ap2d_strides_y)

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

        mp2d_poolsize_y = QLineEdit()
        mp2d_poolsize_y.setValidator(validator)
        mp2d_poolsize_y.setPlaceholderText("Pool Size y value")
        layout.addWidget(mp2d_poolsize_y)

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

        mp2d_strides_y = QLineEdit()
        mp2d_strides_y.setValidator(validator)
        mp2d_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(mp2d_strides_y)

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

        #kernel size
        kernelsize_label = QLabel("Kernel Size: (x, y)")
        layout.addWidget(kernelsize_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2d_kernelsize_x = QLineEdit()
        c2d_kernelsize_x.setValidator(validator)
        c2d_kernelsize_x.setPlaceholderText("Kernel Size x value")
        layout.addWidget(c2d_kernelsize_x)

        c2d_kernelsize_y = QLineEdit()
        c2d_kernelsize_y.setValidator(validator)
        c2d_kernelsize_y.setPlaceholderText("Kernel Size y value")
        layout.addWidget(c2d_kernelsize_y)

        #strides
        strides_label = QLabel("Strides: (x, y)")
        layout.addWidget(strides_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2d_strides_x = QLineEdit()
        c2d_strides_x.setValidator(validator)
        c2d_strides_x.setPlaceholderText("Strides x value")
        layout.addWidget(c2d_strides_x)

        c2d_strides_y = QLineEdit()
        c2d_strides_y.setValidator(validator)
        c2d_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(c2d_strides_y)

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

        #dialation rate
        dialationrate_label = QLabel("Dialation Rate: (x, y)")
        layout.addWidget(dialationrate_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2d_dialationrate_x = QLineEdit()
        c2d_dialationrate_x.setValidator(validator)
        c2d_dialationrate_x.setPlaceholderText("Dialation Rate x value")
        layout.addWidget(c2d_dialationrate_x)

        c2d_dialationrate_y = QLineEdit()
        c2d_dialationrate_y.setValidator(validator)
        c2d_dialationrate_y.setPlaceholderText("Dialation Rate y value")
        layout.addWidget(c2d_dialationrate_y)

        #groups
        groups_label = QLabel("Groups:")
        layout.addWidget(groups_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY
        c2d_groups = QLineEdit()
        c2d_groups.setPlaceholderText("Groups")
        c2d_groups.setValidator(validator)
        layout.addWidget(c2d_groups)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        c2d_activation = QComboBox()
        c2d_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(c2d_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(kernel_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

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

        #kernel size
        kernelsize_label = QLabel("Kernel Size: (x, y)")
        layout.addWidget(kernelsize_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2dt_kernelsize_x = QLineEdit()
        c2dt_kernelsize_x.setValidator(validator)
        c2dt_kernelsize_x.setPlaceholderText("Kernel Size x value")
        layout.addWidget(c2dt_kernelsize_x)

        c2dt_kernelsize_y = QLineEdit()
        c2dt_kernelsize_y.setValidator(validator)
        c2dt_kernelsize_y.setPlaceholderText("Kernel Size y value")
        layout.addWidget(c2dt_kernelsize_y)

        #strides
        strides_label = QLabel("Strides: (x, y)")
        layout.addWidget(strides_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2dt_strides_x = QLineEdit()
        c2dt_strides_x.setValidator(validator)
        c2dt_strides_x.setPlaceholderText("Strides x value")
        layout.addWidget(c2dt_strides_x)

        c2dt_strides_y = QLineEdit()
        c2dt_strides_y.setValidator(validator)
        c2dt_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(c2dt_strides_y)

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

        #dialation rate
        dialationrate_label = QLabel("Dialation Rate: (x, y)")
        layout.addWidget(dialationrate_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2dt_dialationrate_x = QLineEdit()
        c2dt_dialationrate_x.setValidator(validator)
        c2dt_dialationrate_x.setPlaceholderText("Dialation Rate x value")
        layout.addWidget(c2dt_dialationrate_x)

        c2dt_dialationrate_y = QLineEdit()
        c2dt_dialationrate_y.setValidator(validator)
        c2dt_dialationrate_y.setPlaceholderText("Dialation Rate y value")
        layout.addWidget(c2dt_dialationrate_y)

        #groups
        groups_label = QLabel("Groups:")
        layout.addWidget(groups_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY
        c2dt_groups = QLineEdit()
        c2dt_groups.setPlaceholderText("Groups")
        c2dt_groups.setValidator(validator)
        layout.addWidget(c2dt_groups)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        c2dt_activation = QComboBox()
        c2dt_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(c2dt_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(kernel_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)



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

        #kernel size
        kernelsize_label = QLabel("Kernel Size: (x, y)")
        layout.addWidget(kernelsize_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        dc2d_kernelsize_x = QLineEdit()
        dc2d_kernelsize_x.setValidator(validator)
        dc2d_kernelsize_x.setPlaceholderText("Kernel Size x value")
        layout.addWidget(dc2d_kernelsize_x)

        dc2d_kernelsize_y = QLineEdit()
        dc2d_kernelsize_y.setValidator(validator)
        dc2d_kernelsize_y.setPlaceholderText("Kernel Size y value")
        layout.addWidget(dc2d_kernelsize_y)

        #strides
        strides_label = QLabel("Strides: (x, y)")
        layout.addWidget(strides_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        dc2d_strides_x = QLineEdit()
        dc2d_strides_x.setValidator(validator)
        dc2d_strides_x.setPlaceholderText("Strides x value")
        layout.addWidget(dc2d_strides_x)

        dc2d_strides_y = QLineEdit()
        dc2d_strides_y.setValidator(validator)
        dc2d_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(dc2d_strides_y)

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

        #dialation rate
        dialationrate_label = QLabel("Dialation Rate: (x, y)")
        layout.addWidget(dialationrate_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        dc2d_dialationrate_x = QLineEdit()
        dc2d_dialationrate_x.setValidator(validator)
        dc2d_dialationrate_x.setPlaceholderText("Dialation Rate x value")
        layout.addWidget(dc2d_dialationrate_x)

        dc2d_dialationrate_y = QLineEdit()
        dc2d_dialationrate_y.setValidator(validator)
        dc2d_dialationrate_y.setPlaceholderText("Dialation Rate y value")
        layout.addWidget(dc2d_dialationrate_y)

        #depth multiplier
        depthmultiplier_label = QLabel("Depth Multiplier:")
        layout.addWidget(depthmultiplier_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY
        dc2d_depthmultiplier = QLineEdit()
        dc2d_depthmultiplier.setPlaceholderText("Depth Multiplier")
        dc2d_depthmultiplier.setValidator(validator)
        layout.addWidget(dc2d_depthmultiplier)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        dc2d_activation = QComboBox()
        dc2d_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(dc2d_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        #depthwise initializer
        d_initializer_label = QLabel("Depthwise Initializer:")
        layout.addWidget(d_initializer_label)
        depthwise_initializer = QComboBox()
        depthwise_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(depthwise_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

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

        #kernel size
        kernelsize_label = QLabel("Kernel Size: (x, y)")
        layout.addWidget(kernelsize_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        sc2d_kernelsize_x = QLineEdit()
        sc2d_kernelsize_x.setValidator(validator)
        sc2d_kernelsize_x.setPlaceholderText("Kernel Size x value")
        layout.addWidget(sc2d_kernelsize_x)

        sc2d_kernelsize_y = QLineEdit()
        sc2d_kernelsize_y.setValidator(validator)
        sc2d_kernelsize_y.setPlaceholderText("Kernel Size y value")
        layout.addWidget(sc2d_kernelsize_y)

        #strides
        strides_label = QLabel("Strides: (x, y)")
        layout.addWidget(strides_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        sc2d_strides_x = QLineEdit()
        sc2d_strides_x.setValidator(validator)
        sc2d_strides_x.setPlaceholderText("Strides x value")
        layout.addWidget(sc2d_strides_x)

        sc2d_strides_y = QLineEdit()
        sc2d_strides_y.setValidator(validator)
        sc2d_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(sc2d_strides_y)

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

        #dialation rate
        dialationrate_label = QLabel("Dialation Rate: (x, y)")
        layout.addWidget(dialationrate_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        sc2d_dialationrate_x = QLineEdit()
        sc2d_dialationrate_x.setValidator(validator)
        sc2d_dialationrate_x.setPlaceholderText("Dialation Rate x value")
        layout.addWidget(sc2d_dialationrate_x)

        sc2d_dialationrate_y = QLineEdit()
        sc2d_dialationrate_y.setValidator(validator)
        sc2d_dialationrate_y.setPlaceholderText("Dialation Rate y value")
        layout.addWidget(sc2d_dialationrate_y)

        #depth multiplier
        depthmultiplier_label = QLabel("Depth Multiplier:")
        layout.addWidget(depthmultiplier_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY
        sc2d_depthmultiplier = QLineEdit()
        sc2d_depthmultiplier.setPlaceholderText("Depth Multiplier")
        sc2d_depthmultiplier.setValidator(validator)
        layout.addWidget(sc2d_depthmultiplier)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        sc2d_activation = QComboBox()
        sc2d_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(sc2d_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        #depthwise initializer
        d_initializer_label = QLabel("Depthwise Initializer:")
        layout.addWidget(d_initializer_label)
        depthwise_initializer = QComboBox()
        depthwise_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(depthwise_initializer)

        #pointwise initializer
        p_initializer_label = QLabel("Pointwise Initializer:")
        layout.addWidget(p_initializer_label)
        pointwise_initializer = QComboBox()
        pointwise_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(pointwise_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)


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

        #kernel size
        kernelsize_label = QLabel("Kernel Size: (x, y)")
        layout.addWidget(kernelsize_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        clstm2d_kernelsize_x = QLineEdit()
        clstm2d_kernelsize_x.setValidator(validator)
        clstm2d_kernelsize_x.setPlaceholderText("Kernel Size x value")
        layout.addWidget(clstm2d_kernelsize_x)

        clstm2d_kernelsize_y = QLineEdit()
        clstm2d_kernelsize_y.setValidator(validator)
        clstm2d_kernelsize_y.setPlaceholderText("Kernel Size y value")
        layout.addWidget(clstm2d_kernelsize_y)

        #strides
        strides_label = QLabel("Strides: (x, y)")
        layout.addWidget(strides_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        clstm2d_strides_x = QLineEdit()
        clstm2d_strides_x.setValidator(validator)
        clstm2d_strides_x.setPlaceholderText("Strides x value")
        layout.addWidget(clstm2d_strides_x)

        clstm2d_strides_y = QLineEdit()
        clstm2d_strides_y.setValidator(validator)
        clstm2d_strides_y.setPlaceholderText("Strides y value")
        layout.addWidget(clstm2d_strides_y)

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

        #dialation rate
        dialationrate_label = QLabel("Dialation Rate: (x, y)")
        layout.addWidget(dialationrate_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        clstm2d_dialationrate_x = QLineEdit()
        clstm2d_dialationrate_x.setValidator(validator)
        clstm2d_dialationrate_x.setPlaceholderText("Dialation Rate x value")
        layout.addWidget(clstm2d_dialationrate_x)

        clstm2d_dialationrate_y = QLineEdit()
        clstm2d_dialationrate_y.setValidator(validator)
        clstm2d_dialationrate_y.setPlaceholderText("Dialation Rate y value")
        layout.addWidget(clstm2d_dialationrate_y)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        clstm2d_activation = QComboBox()
        clstm2d_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(clstm2d_activation)

        #recurrent activation dropdown
        recurrent_activation_label = QLabel("Recurrent Activation Type:")
        layout.addWidget(recurrent_activation_label)
        clstm2d_recurrent_activation = QComboBox()
        clstm2d_recurrent_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(clstm2d_recurrent_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("Recurrent Initializer:")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        #dropout
        dropout_label = QLabel("Dropout:")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        clstm2d_dropout = QLineEdit()
        clstm2d_dropout.setPlaceholderText("Dropout")
        # clstm2d_dropout.setValidator(float_validator)
        layout.addWidget(clstm2d_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("Recurrent Dropout:")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        clstm2d_recurrent_dropout = QLineEdit()
        clstm2d_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        # clstm2d_recurrent_dropout.setValidator(float_validator)
        layout.addWidget(clstm2d_recurrent_dropout)

        #seed
        seed_label = QLabel("Seed:")
        layout.addWidget(seed_label)

        #add text box
        clstm2d_seed = QLineEdit()
        clstm2d_seed.setPlaceholderText("Seed")
        clstm2d_seed.setValidator(validator)
        layout.addWidget(clstm2d_seed)

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

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        srnn_activation = QComboBox()
        srnn_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(srnn_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("Recurrent Initializer:")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        #dropout
        dropout_label = QLabel("Dropout:")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        srnn_dropout = QLineEdit()
        srnn_dropout.setPlaceholderText("Dropout")
        # srnn_dropout.setValidator(float_validator)
        layout.addWidget(srnn_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("Recurrent Dropout:")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        srnn_recurrent_dropout = QLineEdit()
        srnn_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        # srnn_recurrent_dropout.setValidator(float_validator)
        layout.addWidget(srnn_recurrent_dropout)

        #seed
        seed_label = QLabel("Seed:")
        layout.addWidget(seed_label)

        #add text box
        srnn_seed = QLineEdit()
        srnn_seed.setPlaceholderText("Seed")
        srnn_seed.setValidator(validator)
        layout.addWidget(srnn_seed)

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

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        lstm_activation = QComboBox()
        lstm_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(lstm_activation)

        #recurrent activation dropdown
        recurrent_activation_label = QLabel("Recurrent Activation Type:")
        layout.addWidget(recurrent_activation_label)
        lstm_recurrent_activation = QComboBox()
        lstm_recurrent_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(lstm_recurrent_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("Recurrent Initializer:")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        #unit forget bias dropdown
        unit_forget_bias_label = QLabel("Use Bias:")
        layout.addWidget(unit_forget_bias_label)
        unit_forget_bias = QComboBox()
        unit_forget_bias.addItems(["True", "False"])
        layout.addWidget(unit_forget_bias)

        #dropout
        dropout_label = QLabel("Dropout:")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        lstm_dropout = QLineEdit()
        lstm_dropout.setPlaceholderText("Dropout")
        # lstm_dropout.setValidator(float_validator)
        layout.addWidget(lstm_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("Recurrent Dropout:")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        lstm_recurrent_dropout = QLineEdit()
        lstm_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        # lstm_recurrent_dropout.setValidator(float_validator)
        layout.addWidget(lstm_recurrent_dropout)

        #seed
        seed_label = QLabel("Seed:")
        layout.addWidget(seed_label)

        #add text box
        lstm_seed = QLineEdit()
        lstm_seed.setPlaceholderText("Seed")
        lstm_seed.setValidator(validator)
        layout.addWidget(lstm_seed)

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

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        gru_activation = QComboBox()
        gru_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(gru_activation)

        #recurrent activation dropdown
        recurrent_activation_label = QLabel("Recurrent Activation Type:")
        layout.addWidget(recurrent_activation_label)
        gru_recurrent_activation = QComboBox()
        gru_recurrent_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(gru_recurrent_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("Recurrent Initializer:")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        #dropout
        dropout_label = QLabel("Dropout:")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        gru_dropout = QLineEdit()
        gru_dropout.setPlaceholderText("Dropout")
        # gru_dropout.setValidator(float_validator)
        layout.addWidget(gru_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("Recurrent Dropout:")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        gru_recurrent_dropout = QLineEdit()
        gru_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        # lstm_recurrent_dropout.setValidator(float_validator)
        layout.addWidget(gru_recurrent_dropout)

        #seed
        seed_label = QLabel("Seed:")
        layout.addWidget(seed_label)

        #add text box
        gru_seed = QLineEdit()
        gru_seed.setPlaceholderText("Seed")
        gru_seed.setValidator(validator)
        layout.addWidget(gru_seed)

        #reset after dropdown
        reset_after_label = QLabel("Reset After:")
        layout.addWidget(reset_after_label)
        reset_after = QComboBox()
        reset_after.addItems(["True", "False"])
        layout.addWidget(reset_after)

        # Add widgets specific to configuring layer
        pass


class AddLayerDialog(QDialog):
    layer_added = Signal(dict)
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
        dialog.data_collected.connect(self.handleDataCollected) #connect the signal
        dialog.exec()

    def handleDataCollected(self, data):
        self.layer_added.emit(data) #forward the data upwards
        self.accept()

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

        self.model = LayerTableModel() #initialize model

        #Frame testing
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame_layout = QVBoxLayout(frame)

        #set the model for table view
        self.model = LayerTableModel()
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        frame_layout.addWidget(self.table_view)
        main_layout.addWidget(frame)

        # for i in range(1, 7):
        #     tab = QWidget()
        #     layout = QVBoxLayout(tab)
        #     label = QLabel(f"Content of Model {i}", tab)
        #     self.table_view = QTableView(tab)

        #     self.table_view.setModel(self.model) #set the model for the table view
        #     layout.addWidget(self.table_view)

        #     layout.addWidget(label)
        #     tab_widget.addTab(tab, f"Model {i}")

        # # Adding the tab widget to the main layout with stretch factor.
        # main_layout.addWidget(tab_widget, 1)  # Add stretch to make sure it expands

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
        dialog.layer_added.connect(self.updateModelTab) #connects to new signal
        dialog.exec()

    def updateModelTab(self, layer_data):
        #This method will update the QTableView with the new layer data
        # You need a reference to your QTableView here to update it
        self.model.addLayer(layer_data)
        print("Layer data received:", layer_data)  # Debug: Print or use this data to update the table

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
