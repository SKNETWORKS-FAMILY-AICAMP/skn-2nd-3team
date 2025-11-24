from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from optuna import create_study

def GridSearchTuner():
    study = create_study()
    study.optimize()
    return study

def RandomSearchTuner():
    study = create_study()
    study.optimize()
    return study
    
def OptunaTuner():
    study = create_study()
    study.optimize()
    return study