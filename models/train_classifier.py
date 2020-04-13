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



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
