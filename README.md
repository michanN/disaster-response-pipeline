# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Example Usage](#usage)
6. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation <a name="installation"></a>
The only external library necessary to run this code, beyond the Anaconda distribution of Python, is NLTK [stopwords, punkt, wordnet].

## Project Motivation <a name="motivation"></a>
The purpose of this project is to apply data engineering, NLP and machine learning processes to create a model that can analyze and classify disaster message data from Figure Eight. Depending on the classification (multilabel) of the message it could be sent to the appropriate disaster relief agencies.

## File Descriptions <a name="files"></a>
Project structure and file description below: 
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # flask file that runs app

- data
|- disaster_categories.csv  # dataset including all the categories 
|- disaster_messages.csv  # dataset including all the nessages
|- process_data.py # file includes ETL pipeline
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py # train ML model
|- classifier.pkl  # saved model 

- README.md
```

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

## Example Usage<a name="usage"></a>

Phrased used `We have a lot of problem at Delma 75 Avenue Albert Jode, those people need water and food.`

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Credits to Udacity for the project/starter code and Figure Eight for the dataset.

