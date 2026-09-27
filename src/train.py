from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.svm import SVR
import joblib
import numpy as np
from pathlib import Path

from traitment import load_data, find_outliers

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "Models"


def train_model():
    df = load_data()
    _, _, _, _, _, mask = find_outliers(df, "Exam_Score")
    tuned = df[~mask]  # data bla outliers

    X = tuned.drop(columns='Exam_Score')
    Y = tuned["Exam_Score"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10, shuffle=True)

    # vars needed
    numerical_columns = X.select_dtypes(include="number").columns
    nominal_columns = ["Gender", "School_Type", "Extracurricular_Activities",
                       "Internet_Access", "Learning_Disabilities", "Peer_Influence"]
    ordinal_categories = {
        "Parental_Involvement": ["Low", "Medium", "High"],
        "Access_to_Resources": ["Low", "Medium", "High"],
        "Motivation_Level": ["Low", "Medium", "High"],
        "Family_Income": ["Low", "Medium", "High"],
        "Teacher_Quality": ["Low", "Medium", "High"],
        "Parental_Education_Level": ["High School", "College", "Postgraduate"],
        "Distance_from_Home": ["Near", "Moderate", "Far"],
    }

    param_grids = {
        "LinearRegression": {
            "model__fit_intercept": [True, False],
            "model__positive": [False, True],
        },
        "RandomForest": {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        },
        "SVR": {
            "model__C": [0.1, 1, 10, 100],
            "model__kernel": ["linear", "rbf"],
            "model__epsilon": [0.1, 0.5, 1.0],
        },
    }

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(),
        "SVR": SVR(),
    }

    numeric_pipline = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())
    ])

    ordinal_pipline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('encoder', OrdinalEncoder(categories=[ordinal_categories[c] for c in ordinal_categories.keys()]))
    ])

    nominal_pipline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('onehot_encoder', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numeric', numeric_pipline, numerical_columns),
        ('ordinal', ordinal_pipline, list(ordinal_categories.keys())),
        ('nominal', nominal_pipline, nominal_columns),
    ])


    def build_pipeline(model):
        return Pipeline([
            ('preprocessor', clone(preprocessor)),
            ('model', model)
        ])


    result = {}
    best_r2 = float('-inf')
    best_pipe = None
    best_name = None

    for name, model in models.items():
        grid = GridSearchCV(
            build_pipeline(model),
            cv=3,
            param_grid=param_grids[name],
            scoring="r2"
        )
        grid.fit(x_train, y_train)
        predict = grid.predict(x_test)

        result[name] = {
            "mean_r2": grid.best_score_,
            "best_parameters": grid.best_params_,
            "MAE": mean_absolute_error(y_test, predict),
            "RMAE": np.sqrt(mean_squared_error(y_test, predict)),
            "R2": r2_score(y_test, predict),
        }
        if best_r2 < result[name]["R2"]:
            best_pipe = grid.best_estimator_
            best_r2 = result[name]["R2"]
            best_name = name

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_DIR / "best_model.joblib")

    return result


if __name__ == "__main__":
    train_model()
