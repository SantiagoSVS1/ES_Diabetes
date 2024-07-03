from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

'''
This class is used to get the models for the experiment.

Attributes:
    models (dict): A dictionary containing the models.
'''
class Models:
    def __init__(self):
        self.models = {
            "logistic_regression": LogisticRegression(max_iter=1000, tol=1e-4),
            "sgd_classifier": SGDClassifier(max_iter=1000, tol=1e-4),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "ada_boost": AdaBoostClassifier(),
            "svc": SVC(),
            "knn": KNeighborsClassifier(),
            "decision_tree": DecisionTreeClassifier(),
            "naive_bayes": GaussianNB(),
            "lda": LinearDiscriminantAnalysis(),
            "qda": QuadraticDiscriminantAnalysis(),
            # "mlp": MLPClassifier(max_iter=1000, tol=1e-4)
        }

    def get_model(self, model_name):

        # Return the model with the specified name
        return self.models.get(model_name, None)

    def get_all_models(self):

        # Return all the models
        return self.models