from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class RanForestModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.param_grid = {'max_features': ['sqrt', None],
                            'max_depth': [10, 20, 30, 40,50],
                            'min_samples_split': [2, 5, 10, 15],
                            'min_samples_leaf': [1, 2, 4, 8]}
        self.best_model = None
        self.model_his = []

    def train(self,X,y):

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        try:
            RF_grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=5, n_jobs=-1).fit(X_train, y_train)
            self.best_model = RF_grid_search.best_estimator_
        except Exception as e:
            print("Lỗi khi training:", e)
            self.best_model = self.model
        
        #self.best_model = RF_grid_search.best_estimator_
        y_pred = self.best_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        self.model_his.append({'model':  self.best_model, 'accuracy': accuracy})
        if len(self.model_his) > 3:
            self.model_his.pop(0) 
        return accuracy
    
    def get_param_his(self):
        return self.model_his
    
    def setModel(self,i):
        self.model = self.model_his[i]['model']
    
    def predict(self,data):
        y_pre = self.model.predict(data)
        return y_pre






