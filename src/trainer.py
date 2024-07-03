from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle

class ModelTrainer:
    def __init__(self, models, X_train, y_train):
        self.models = models
        self.X_train = X_train
        self.y_train = y_train
        self.results = {}
        
    def train(self):        
        print("\nTraining models...")
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            model_path = f"data/models/{model_name}.pkl"
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
        print("\nTraining complete.\n")
        return self.models
    
    def optimize_hyperparameters(self, model_name, X_train, y_train, param_grid, search_method='grid', n_iter=10):
        
        model = self.models[model_name]
        if search_method == 'grid':
            search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
        else:
            search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=5, n_jobs=-1)
        
        search.fit(X_train, y_train)
        best_params = search.best_params_
        
        self.models[model_name] = search.best_estimator_
        
        print(f"Best parameters for {model_name}: {best_params}")