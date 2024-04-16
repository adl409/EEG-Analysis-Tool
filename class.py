class Model:
    def __init__(self, input_size):
        self.input_size = input_size
        self.model = self.build_model()
        self.compile_model()
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_size,))
        ])
        return model
    
    def compile_model(self):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
    
    def add_dense_layer(self, units, activation='relu'):
        self.model.add(tf.keras.layers.Dense(units, activation=activation))
    
    def add_dropout_layer(self, rate):
        self.model.add(tf.keras.layers.Dropout(rate))
    
    def change_input_size(self, input_size):
        self.input_size = input_size
        self.model = self.build_model()
        self.compile_model()
    
    def train(self, train_data, train_labels):
        self.model.fit(train_data, train_labels, epochs=10)
    
    def evaluate(self, test_data, test_labels):
        return self.model.evaluate(test_data, test_labels)
