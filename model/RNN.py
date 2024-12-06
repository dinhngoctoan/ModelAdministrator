import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense # type: ignore
from sklearn.metrics import accuracy_score

class RNNModel:
    def __init__(self, X,y):
        X = X.values.reshape((X.shape[0], 1, X.shape[1]))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = X_train, X_val, y_train, y_val
        self.model = self.build_model(X_train)
        self.param_his = []

    def build_model(self,X_train):
        sequence_length = X_train.shape[1]
        num_features = X_train.shape[2]
        model = Sequential()
        model.add(SimpleRNN(64, input_shape=(sequence_length, num_features), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  

        # Biên dịch mô hình
        model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
        return model 
    
    def train(self,X,y):
        X = X.values.reshape((X.shape[0], 1, X.shape[1]))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = X_train, X_val, y_train, y_val
        self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32)
        y_pred_prob = self.model.predict(self.X_val)
        y_pred = (y_pred_prob > 0.5).astype("int32")
        accuracy = accuracy_score(self.y_val, y_pred)
        weights = self.model.get_weights()  # Lấy trọng số của model
        self.param_his.append({'weights': weights, 'accuracy': accuracy})
        if len(self.param_his) > 3:
            self.param_his.pop(0)
        return accuracy
    
    def get_param_his(self):
        return self.param_his

    def setModel(self,index):
        self.model.set_weights(self.param_his[index]['weights'])
    
    def predict(self,data):
        y_pre = self.model.predict(data)
        return y_pre
