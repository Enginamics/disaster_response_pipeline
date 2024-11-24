"""
train_classifier.py

- Python script for training and saving a machine learning model to classify disaster response messages. 
- It saves the trained model as a pickle file for the web application.

Usage:
    python train_classifier.py <database_filepath> <model_filepath>

    - <database_filepath>: Path to the sqlite database containing the preprocessed data.
    - <model_filepath>: Path to save the model as a pickle file.

Example:
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
"""

# --------------------
# Imports:
# --------------------
# Standard Library Imports:
import sys
import os
import re
import pickle

# Third party imports:
import pandas as pd
import numpy as np

# Import data from SQL
from sqlalchemy import create_engine

# Add the project root to Python path to be able to use local imports
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_PATH)

# local Imports
from tools.tokenizer import tokenize
from tools.score import multioutput_f1_score
from tools.transformer import TextFeatureExtractor

# Scikit-learn for Machine Learning Pipeline and Model Evaluation
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, make_scorer

# --------------------
# Functions
# --------------------

def load_data(database_filepath):
    """
    Load data from sqlite database and prepare features and targets.

    Args:
        database_filepath (str): Path to the sqlite database.

    Returns:
        X (pd.Series): The feature column (in this case: messages).
        Y (pd.DataFrame): Target variables (in this case: categories).
        category_names (list): Names of the target variable categories.
    """
    try:
        # Connect to the sqlite database
        engine = create_engine(f'sqlite:///{database_filepath}')
        
        # Load table into a DataFrame
        df = pd.read_sql_table('messages_categories', engine)
        
        # Extract features and targets
        X = df['message']
        Y = df.iloc[:, 4:]  # Adjust this index range based on your database structure
      
        # Get the category names
        category_names = Y.columns.tolist()
        
        return X, Y, category_names

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def build_model():
    """
    Build a machine learning pipeline with a text processing pipeline 
    and a multi-output classifier using XGBoost, and integrate GridSearchCV 
    for optimization using a custom scoring function for multi-output F1-score.

    Returns:
        GridSearchCV: A grid search model with pipeline and hyperparameter tuning.
    """

    # Define the machine learning pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
                ('tfidf', TfidfTransformer())
            ])),
            ('custom_features', TextFeatureExtractor()),  # Add custom features here
        ])),
        ('clf', MultiOutputClassifier(XGBClassifier(
            random_state=42, 
            scale_pos_weight=1,
            eval_metric='logloss'
        )))
    ])

    # Define hyperparameters for GridSearchCV
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__max_depth': [3, 5],
        'clf__estimator__learning_rate': [0.1, 0.3],
        'clf__estimator__scale_pos_weight': [1, 5, 10],
    }

    # Custom scorer for GridSearchCV
    scorer = make_scorer(multioutput_f1_score)

    # Grid search with cross-validation
    model = GridSearchCV(
        pipeline,
        param_grid=parameters,
        scoring=scorer,  # Use custom scorer
        verbose=3,  # Display progress logs
        n_jobs=-1,  # Use all CPU cores
        cv=3  # 3-fold cross-validation
    )

    # Return GridSearch object
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of a trained model or a GridSearchCV model on the test set.

    Args:
        model: Trained model or GridSearchCV object.
        X_test (pd.DataFrame): Test features.
        Y_test (pd.DataFrame): True labels for the test set.
        category_names (list): List of category names.

    Returns:
        None
    """
    # If the model is a GridSearchCV object, use the best estimator
    if hasattr(model, 'best_estimator_'):
        print("Evaluating the best model from GridSearchCV...")
        best_model = model.best_estimator_
        best_params = model.best_params_
        print("Best Parameters:", best_params)
    else:
        print("Evaluating the provided model...")
        best_model = model  # Use the model directly

    # Predict the labels for the test set
    Y_pred = best_model.predict(X_test)

    # Initialize lists to collect average metrics
    avg_precision, avg_recall, avg_f1 = [], [], []

    # Iterate over each category and compute metrics
    for i, category in enumerate(category_names):
        print(f"\nCategory: {category}")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        print(f"Accuracy: {accuracy_score(Y_test.iloc[:, i], Y_pred[:, i]):.2f}")

        # Collect metrics for averages
        report = classification_report(
            Y_test.iloc[:, i], Y_pred[:, i], output_dict=True
        )
        avg_precision.append(report["weighted avg"]["precision"])
        avg_recall.append(report["weighted avg"]["recall"])
        avg_f1.append(report["weighted avg"]["f1-score"])

    # Display average metrics across all categories
    print("\n--- Overall Metrics ---")
    print(f"Average Precision: {np.mean(avg_precision):.2f}")
    print(f"Average Recall: {np.mean(avg_recall):.2f}")
    print(f"Average F1 Score: {np.mean(avg_f1):.2f}")


def save_model(model, model_filepath):
    """
    Save the trained model to a pickle file.

    Args:
        model: Trained machine learning model.
        model_filepath (str): Filepath to save the model as a pickle file.

    Returns:
        None
    """
    try:
        # Save the model to the specified filepath
        with open(model_filepath, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model successfully saved to {model_filepath}")

    except Exception as e:
        print(f"Error saving model: {e}")


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
     
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()