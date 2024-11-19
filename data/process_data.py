"""
process_data.py

- Python script which serves as the ETL pipeline for the project. 
- It loads the message and category datasets, cleans and transforms the data
- It saves the processed data into an sqlite database for analysis within the machine learning pipeline.

Usage:
    python process_data.py <messages_filepath> <categories_filepath> <database_filepath>
"""

# Import libraries for ETL pipeline
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
    messages_filepath (str): File path of the messages CSV file.
    categories_filepath (str): File path of the categories CSV file.

    Returns:
    pd.DataFrame: Merged DataFrame which includes messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    """
    Cleaning the merged DataFrame by splitting categories and converting values.

    Args:
    df (pd.DataFrame): Merged DataFrame with messages and categories.

    Returns:
    pd.DataFrame: Cleaned DataFrame with categories separated into multiple columns.
    """
    # Create DataFrame of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    # Convert category values to numeric values (0 or 1)
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
          # Ensure "binary values" (0 or 1):
        categories[column] = categories[column].apply(lambda x: 1 if x > 1 else x)
    
    # Drop the original categories column from df and concatenate the new categories DataFrame
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # Dropping duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Save the cleaned up dataset into a sqlite database.

    Args:
    df (pd.DataFrame): Cleaned DataFrame.
    database_filename (str): File path of the sqlite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages_categories', engine, index=False, if_exists='replace')


def main():
    """
    Run the ETL pipeline: Load data, clean data and save to database.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print(
            'Please provide the filepaths of the messages and categories '
            'datasets as the first and second argument respectively, as '
            'well as the filepath of the database to save the cleaned data '
            'to as the third argument. \n\nExample: python process_data.py '
            'disaster_messages.csv disaster_categories.csv DisasterResponse.db'
        )


if __name__ == '__main__':
    main()
