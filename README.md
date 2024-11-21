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
- textblob==0.18.0.post0   
- xgboost==2.1.2   

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
    - *.csv: raw data provided by data science course   
    - DisasterResponse.db: sqlite database ("cleaned data") used for training   
    - **process_data.py**: Python script which realizes the ETL pipeline for the project.   

- ./models   
    - classifier.pkl: pkl file of trained model   
    - **train_classifier.py**: Python script which realizes the ML pipeline for the project   

- ./tools   
    This folder includes common tools / utilities, which can be used by all other scripts of this project.
    - **score.py**: Python script with custom scorer   
    - **tokenizer.py**: Python script used to tokenize text   
    - **transformer.py**: Python script used to transform text   

- requirements.txt   
    - This file specifies the dependencies used by the project (see chapter [Installing](#installing) for more details) 

## Installing

Ensure you have [Python 3.12.5](https://www.python.org/downloads/release/python-3125/) or a newer version installed.

1. Clone this repository:
    ```bash
    git clone https://github.com/Enginamics/disaster_response_pipeline.git
    ```
2. Navigate to the project directory:
    ```bash
    cd /path/to/this/repository
    ```
3. Create a virtual environment:
    ```bash
    python -m venv .venv
    ```
4. Activate the virtual environment:
    - On **Windows**:
        ```bash
        .venv\Scripts\activate
        ```
    - On **macOS/Linux**:
        ```bash
        source .venv/bin/activate
        ```
5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Instructions

1. **Set Up the Environment**:
   Follow the steps in the [Installing](#installing) section to prepare the environment.

2. **Run ETL and ML Pipelines**:
   Execute the following commands from the project’s root directory:
   - To clean data and store it in a database:
     ```bash
     python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
     ```
   - To train the classifier and save the model:
     ```bash
     python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
     ```

3. **Run the Web Application**:
   Navigate to the `app/` directory and start the web application:
   ```bash
   cd ./app
   python run.py
   ```

4. **Go to** http://127.0.0.1:3001

## Licensing, Authors, and Acknowledgements

A special thanks to the [Udacity - Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025) course, which made this analysis possible.

## Troubleshooting
