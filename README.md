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
#### To run ETL pipeline that cleans data
##### python train_classifier.py '../data/categories_table.db' './modelwww.pkl'

### 2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:<br>

Loads data from the SQLite database<br>
Splits the dataset into training and test sets<br>
Builds a text processing and machine learning pipeline<br>
Trains and tunes a model using GridSearchCV<br>
Outputs results on the test set<br>
Exports the final model as a pickle file<br>
#### To run ML pipeline that trains classifier
#####  python train_classifier.py '../data/categories_table.db' './modelwww.pkl'  <br>


### 3. Flask Web App
Add data visualizations using Plotly in the web app.<br>
#### run APP
#####  python run.py  


### Files
  * ./app/templates/go.html      web of file display  
  * ./app/templates/master.html   web of file display  
  * ./app/templates/run.py   run web of page  <br>

  * ./data/categories.csv   categories data  
  * ./data/messages.csv     messages data  
  * ./data/process_data.py    process categories and messages data and save to SQLite databases   <br>

  * ./models/train_classifier.py   use pipline build model    <br>
