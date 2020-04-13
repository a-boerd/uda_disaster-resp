import sys
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet'])
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle

def load_data(database_filepath):
    """
    input:
    database_filepath: loading data from the given location
    output:
    returning X,y and categories of y
    """
    #load data from db
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("disaster_message_cat",con=engine)
    #create X and y
    X = df["message"]
    y = df.drop(["message","genre","id","original"],axis=1)
    category_names=y.columns
    return X,y,category_names

def tokenize(text):
    """
    input: 
    text that will be tokenized
    output:
    clean tokens of the given text
    """
    #create tokens
    tokens = word_tokenize(text)
    #create lemmatizer
    lemmatizer = WordNetLemmatizer()
    #clean tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
def build_model():
    """
    setting up the model (pipeline and gridsearch for optimizing parameters)
    output: model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    #provide a set of parameters for gridsearch
    parameters = {
    'clf__estimator__min_samples_split':[2,4,6],
    'vect__ngram_range': ((1, 1), (1, 2)),
    'tfidf__use_idf': (True, False)
    }
    cv = GridSearchCV(pipeline,param_grid=parameters, n_jobs=-1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    printing results 
    input:
    model: model that is being evaluated
    X_test: tokens of test-dataset
    Y_test: results of test-dataset
    category_names: categories/columns that results will be provided for

    """
    y_pred=model.predict(X_test)
    y_pred_df=pd.DataFrame(data=y_pred,columns=Y_test.columns)
    for col in Y_test.columns:
        print(col,classification_report(y_true=Y_test[col],y_pred=y_pred_df[col]))


def save_model(model, model_filepath):
    """
    input:
    model: model to be saved
    model_filepath: path to save to
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
