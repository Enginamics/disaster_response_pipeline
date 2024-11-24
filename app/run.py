# --------------------
# Imports
# --------------------
# Standard Library Imports
import json
import sys
import os

# Adding project root to Python path to be able to use local imports
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_PATH)

# Third-Party Library Imports
import joblib
import pandas as pd
import plotly
from flask import Flask, render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

# --------------------
# Constants
# --------------------
DATABASE_PATH = os.path.join(ROOT_PATH, "data", "DisasterResponse.db")
MODEL_PATH = os.path.join(ROOT_PATH, "models", "classifier.pkl")

# --------------------
# App Initialization
# --------------------
app = Flask(__name__)

# --------------------
# Load Data and Model
# --------------------
engine = create_engine(f"sqlite:///{DATABASE_PATH}")
df = pd.read_sql_table("messages_categories", engine)

model = joblib.load(MODEL_PATH)


# --------------------
# Routes
# --------------------
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Main page route: Displays visuals and input text for model prediction.
    """
    # Extract data needed for visuals
    # Original data for genre distribution
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Data for category distribution
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    # Data for genre proportions
    genre_proportions = df['genre'].value_counts(normalize=True) * 100
    genre_labels = list(genre_proportions.index)

    # Data for example messages
    example_messages = df[['message', 'genre']].sample(5).values.tolist()

    # Create visuals
    graphs = [
        # Example Messages Visualization
        {
            'data': [
                {
                    "type": "table",
                    "header": {
                        "values": ["Message", "Genre"],
                        "align": "left"
                    },
                    "cells": {
                        "values": list(zip(*example_messages)),
                        "align": "left",
                        "format": ""
                    }
                }
            ],
            'layout': {
                'title': 'Five randomly selected Messages from the Dataset',
                'margin': {"t": 30, "l": 0, "r": 0, "b": 0}
            }
        },
        # Distribution of Message Genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        # Distribution of Message Categories
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Categories"}
            }
        },
        # Proportion of Messages by Genre
        {
            'data': [
                {
                    "type": "pie",
                    "labels": genre_labels,
                    "values": genre_proportions.tolist(),
                    "hoverinfo": "label+percent",
                    "textinfo": "percent"
                }
            ],
            'layout': {
                'title': 'Proportion of Messages by Genre'
            }
        }
    ]

    # JSON encoding for rendering in HTML
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render the `master.html` template with the updated graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


# --------------------
# Main Function
# --------------------
def main():
    """
    Main function - run Flask app.
    """
    app.run(host="0.0.0.0", port=3001, debug=True)


# Entry point for the script
if __name__ == "__main__":
    main()