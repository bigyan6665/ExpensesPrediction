from src.exception import CustomException
import os, pickle, sys
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    report = {}
    for k, v in models.items():
        # k = model name, v = model obj
        par = params[k]
        # print(par)
        gs = GridSearchCV(v, par, cv=3)
        gs.fit(x_train, y_train)
        # print(gs.best_params_)

        v.set_params(**gs.best_params_)
        model = v.fit(x_train, y_train)
        # print(model.get_params())

        y_test_pred = model.predict(x_test)
        r2score = r2_score(y_test, y_test_pred)
        report[r2score] = [k, model]

    return report  # report = dict having model name as key and [model obj,its accuracy] as value


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# takes report and returns the name and obj of the model having best r2score
def find_best_model(report):
    max_acc = max(report.keys())
    best_model_name, best_model_obj = report[max_acc]
    return (best_model_name, best_model_obj, max_acc)
