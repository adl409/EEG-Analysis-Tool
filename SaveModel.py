def saveModel(self):
        if self.model is not None:
            file_dialog = QFileDialog(self)
            file_dialog.setNameFilter("HDF5 files (*.h5)")
            file_dialog.setWindowTitle("Save Model File")
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            if file_dialog.exec_():
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    model_path = selected_files[0]
                    try:
                        self.model.save(model_path)
                        self.status_label.setText("Model saved successfully")
                    except Exception as e:
                        self.status_label.setText("Failed to save model. Please try again.")
        else:
            self.status_label.setText("No model to save")
