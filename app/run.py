# --------------------
# Imports
# --------------------
# Standard Library Imports
import json
import sys
import os

# Add the project root to Python path to be able to use local imports
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_PATH)

# local Imports
from tools.tokenizer import tokenize
from tools.score import multioutput_f1_score

# Third-Party Library Imports
import joblib
import pandas as pd
import plotly
from flask import Flask, jsonify, render_template, request
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
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
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
    Main function to run the Flask app.
    """
    app.run(host="0.0.0.0", port=3001, debug=True)


# Entry point for the script
if __name__ == "__main__":
    main()