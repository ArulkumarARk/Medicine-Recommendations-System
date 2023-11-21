import os  # Import os module for file paths
import pickle
import re
import gzip
import warnings

from flask import Flask, render_template, request
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)


# Function to load the saved model
def load_model(model_filename):
    with gzip.open(model_filename, 'rb') as model_file:
        cv, similarity, df = pickle.load(model_file)
    return cv, similarity, df


# Function to preprocess input and get recommendations for both Allopathy and Ayurveda
def recommend_medicines(input_text, allopathy_cv, allopathy_similarity, allopathy_df, ayurveda_cv, ayurveda_similarity,
                        ayurveda_df):
    ps = PorterStemmer()
    keywords = [ps.stem(word.lower()) for word in re.findall(r'\b\w+\b', input_text)]

    # Allopathy recommendations
    allopathy_input_vector = allopathy_cv.transform([" ".join(keywords)]).toarray()
    allopathy_input_similarity = cosine_similarity(allopathy_input_vector,
                                                   allopathy_cv.transform(allopathy_df['tags']).toarray())

    allopathy_recommendations = []
    for i in range(3):
        index = allopathy_input_similarity.argsort()[0][-i - 2]
        allopathy_recommendations.append(allopathy_df.iloc[index]['Drug_Name'])

    # Remove duplicates from the list
    unique_allopathy_recommendations = list(set(allopathy_recommendations))

    # Ayurveda recommendations
    ayurveda_input_vector = ayurveda_cv.transform([" ".join(keywords)]).toarray()
    ayurveda_input_similarity = cosine_similarity(ayurveda_input_vector,
                                                  ayurveda_cv.transform(ayurveda_df['tags']).toarray())

    ayurveda_recommendations = []
    for i in range(3):
        index = ayurveda_input_similarity.argsort()[0][-i - 2]
        ayurveda_recommendations.append(ayurveda_df.iloc[index]['drug'])

    # Remove duplicates from the list
    unique_ayurveda_recommendations = list(set(ayurveda_recommendations))

    return unique_allopathy_recommendations, unique_ayurveda_recommendations


# Load Allopathy model
allopathy_cv, allopathy_similarity, allopathy_df = load_model('allopathy_cosine_similarity_model.pkl.gz')

# Load Ayurveda model
ayurveda_cv, ayurveda_similarity, ayurveda_df = load_model('ayurvedic_cosine_similarity_model.pkl.gz')


# Flask route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Flask route to handle form submission
@app.route('/recommendations', methods=['POST'])
def recommendations():
    if request.method == 'POST':
        user_input_text = request.form['symptoms']
        allopathy_recommendations, ayurveda_recommendations = recommend_medicines(
            user_input_text, allopathy_cv, allopathy_similarity, allopathy_df, ayurveda_cv, ayurveda_similarity, ayurveda_df
        )
        return render_template('recommendations.html',
                               allopathy_recommendations=allopathy_recommendations,
                               ayurveda_recommendations=ayurveda_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
