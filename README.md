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

3. Go to http://view6914b2f4-3001.udacity-student-workspaces.com/

## Project Motivation<a name="motivation"></a>

For this project, I was interestested in using Stack Overflow data from 2017 and focusing on the behavioral/social side:

1. What motivates people to update their resumes?
2. How competitive are users of Stack Overflow (towards themselves and others) and how do these attributes correlate?
3. What are the main methods people use to teach themselves new knowledge? (left this out in the blog-post because it wasn't too spectacular, whoever I was curious about it)
4. What's people's take on working with other people's code and can we predict this by looking at other information we have?



## File Descriptions <a name="files"></a>

There is one notebook available here to showcase work related to the above questions.  It first shows the exploratory part of the analysis as well as the visualization and ends up with building the randomforest model for predicting people's answers to the "working with other people's code" question.  Markdown cells and comments in the code cells were used to assist in walking through the thought process for individual steps.  

This is the data-basis for the analysis that was conducted (2017 stackoverflow survey): https://drive.google.com/uc?export=download&id=0B6ZlG_Eygdj-c1kzcmUxN05VUXM


## Results<a name="results"></a>

The main findings of the analysis can be found at the post available [here](https://medium.com/@boerdolf/how-to-find-your-perfect-match-in-all-the-stackoverflow-users-f567fd978213).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit to Stack Overflow for the data.  Info on Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/stackoverflow/so-survey-2017/data). Also, credit to https://github.com/jjrunner/stackoverflow for the layout of this readme file.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://view6914b2f4-3001.udacity-student-workspaces.com/
