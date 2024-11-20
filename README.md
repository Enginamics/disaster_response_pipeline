# Disaster Response Pipeline Project

This is a repository for the [Udacity - Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025) Course 4 - Data Engineering project.

For the analysis the following Python version and packages were used (see chapter [Installing](#installing) for more details) 

**Python Version:**   
- Python 3.12.5

**Packages:**   
- flask==3.1.0   
- nltk==3.9.1   
- numpy==2.1.3   
- pandas==2.2.3   
- plotly==5.24.1   
- scikit-learn==1.5.2   
- sqlalchemy==2.0.36   

## Table of Contents

1. [Project Motivation](#project-motivation)
2. [Results](#results)
3. [Files and Folders](#files-and-folders)
4. [Installing](#installing)
5. [Instructions](#instructions)
6. [Licensing, Authors, and Acknowledgements](#licensing-authors-and-acknowledgements)
7. [Troubleshooting](#troubleshooting)

## Project Motivation

The main objective of this project is to build a working ETL (Extract, Transform, and Load) and Machine Learning Pipeline on real world data.

## Results

## Files and Folders

This GitHub repository consists of the following main files and folders:

- ./app
    - ./templates: folder which includes the html templates for the web application   
    - **run.py**: Python script which starts and runs the web application

- ./data
    - **process_data.py**: Python script which realizes the ETL pipeline for the project. 

- ./models
    - **train_classifier.py**: Python script which realizes the ML pipeline for the project

- ./tools
    - **tokenizer.py**: Python script used to tokenize text

- requirements.txt   
    - This file specifies the dependencies used by the project (see chapter [Installing](#installing) for more details) 

## Installing

Make sure [python 3.12.5](https://www.python.org/downloads/release/python-3125/) (or newer) is installed on your machine

Clone this repository to your machine:
```shell
git clone https://github.com/Enginamics/disaster_response_pipeline.git
```
Then create the virtual python environment in the cloned repository
```shell
cd /path/to/this/repository
```
```shell
python -m venv .venv
```
Then activate the virtual environment
- On Windows:
    ```shell
    .venv\Scripts\activate
    ```
- On macOS/Linux:
    ```shell
    source .venv/bin/activate
    ```
The project uses the python packages

- pandas
- sqlalchemy

which all can be installed via the provided requirements.txt into the virtual environment
```shell
pip install -r requirements.txt
```

## Instructions:

1. Install and activate the virtual python environment

    - see chapter [Installing](#installing) for more details

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

## Licensing, Authors, and Acknowledgements

A special thanks to the [Udacity - Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025) course, which made this analysis possible.

## Troubleshooting

