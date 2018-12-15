import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads message and category datasets.

    INPUT:
        messages_filepath (string): The path to the messages dataset.
        categories_filepath (string): The path to the categories dataset.

    OUTPUT:
        df (dataframe): The merged dataset.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id', how='outer')

    return df


def clean_data(df):
    '''
    Cleans the merges dataframe so it can be used inside the ML pipeline.

    INPUT:
        df (dataframe): The merged dataframe from messages and categories.

    OUTPUT:
        df (dataframe): Cleaned dataframe.
    '''
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    category_colnames = list(row.apply(lambda x: x[:-2]))
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # delete rows where related == 2
    df.query('related != 2', inplace=True)

    df.drop_duplicates(subset='id', inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Saves cleaned df to sqlite database.

    INPUT:
        df (dataframe): The dataframe to save.
        database_filename (string): Path to database.
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
