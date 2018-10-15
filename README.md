### DateScienceDisaster 
# Disaster Pipline Project

## Introduction <br>
In this Project, I will find a data set containing real messages that were sent during disaster events and 
creating a machine learning pipeline to categorize these events so that could send the messages to an appropriate disaster relief agency.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories.

#### Project Components
There are three components you'll need to complete for this project.

### 1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

Loads the messages and categories datasets<br> 
Merges the two datasets<br>
Cleans the data<br>
Stores it in a SQLite database <br>

### 2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:<br>

Loads data from the SQLite database<br>
Splits the dataset into training and test sets<br>
Builds a text processing and machine learning pipeline<br>
Trains and tunes a model using GridSearchCV<br>
Outputs results on the test set<br>
Exports the final model as a pickle file<br>

### 3. Flask Web App
Add data visualizations using Plotly in the web app.<br>
