    def saveLayer(self):
        # Append to main dictionary the layer type the user wants to add
        length = len(nnet.datadict["model_1"]["layers"])

        global index


        if self.layer_type == "dense":
            nnet.datadict["model_1"]["layers"].insert(index,
                Dense(
                    length, 
                    int(float(self.config[0].text())), 
                    self.config[1].currentText(), 
                    True if self.config[2].currentText() == "True" else False, 
                    self.config[3].currentText(),
                    self.config[4].currentText()
                    )
                )
        elif self.layer_type == "flatten":
            nnet.datadict["model_1"]["layers"].insert(index,
                Flatten(
                    length
                    )
                )
        elif self.layer_type == "zero Padding 2d":
            nnet.datadict["model_1"]["layers"].insert(index,
                Zero_Padding_2d(
                    length, 
                    (int(float(self.config[0].text())), int(float(self.config[1].text())))
                    )
                )
        elif self.layer_type == "average Pooling 2d":
            nnet.datadict["model_1"]["layers"].insert(index,
                Average_Pooling_2d(
                    length, 
                    (int(float(self.config[0].text())), int(float(self.config[1].text()))), 
                    (int(float(self.config[2].text())), int(float(self.config[3].text()))), 
                    self.config[4].currentText()
                    )
                )
        elif self.layer_type == "max Pooling 2d":
            nnet.datadict["model_1"]["layers"].insert(index,
                Max_Pool_2d(
                    length, 
                    (int(float(self.config[0].text())), int(float(self.config[1].text()))), 
                    (int(float(self.config[2].text())), int(float(self.config[3].text()))), 
                    self.config[4].currentText()
                    )
                )
        elif self.layer_type == "convolution 2d":
            nnet.datadict["model_1"]["layers"].insert(index,
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
        elif self.layer_type == "convolution 2d Transpose":
            nnet.datadict["model_1"]["layers"].insert(index,
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
        elif self.layer_type == "simplernn":
            nnet.datadict["model_1"]["layers"].insert(index,
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
        elif self.layer_type == "lstm":
            nnet.datadict["model_1"]["layers"].insert(index,
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
        elif self.layer_type == "gru":
            nnet.datadict["model_1"]["layers"].insert(index,
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
        
        del nnet.datadict["model_1"]["layers"][index+1]


        nnet.layoutChanged.emit()
        self.accept()
