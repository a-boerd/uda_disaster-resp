# Disaster Response Pipeline Project

1. [Instructions](#instructions)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://SPACEID-3001.SPACEDOMAIN (where both SPACEID and SPACEDOMAIN need to be replaced with whatever the command env|grep WORK returns)

## Project Motivation<a name="motivation"></a>

For this project, the goal was to create a model for an API that is able to classify disaster messages. This is based on data provided by [Figure Eight](https://www.figure-eight.com/).


## File Descriptions <a name="files"></a>

You will find several different files in this repository:
1. Data folder: You can find the original message data (disaster_messages.csv) and the corresponding categorizations (disaster_categories.csv), both provided by Figure Eight. Also, the process_data.py where the cleaning of the data is being done after which the data is saved to a database-file, which you will also find in there (DisasterResponse.db)
2. Models folder: This is where you can find the train_classifier.py file that trains a classifier based on the provided data.
3. App folder: You will find the file that runs the API (run.py) as well as some html templates in the app/templates folder.

## Results<a name="results"></a>

I did manage to train a ML model that can classify messages that a user can input in the API. Potential for further improvement is certainly in the area of gridsearch, where I had to limit myself to quite a small amount of parameters in order for the calculation time to remain bearable.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit to [Figure Eight](https://www.figure-eight.com/) for the data. 
