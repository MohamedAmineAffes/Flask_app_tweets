from flask import Flask, request, jsonify
from ntscraper import Nitter
from pymongo import MongoClient
import re
import pandas as pd
import json
from detoxify import Detoxify
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
import numpy as np


app = Flask(__name__)

#Initialize scraper library
scraper = Nitter()

def get_tweets(query, mode, size, instance_url):
  tweets = scraper.get_tweets(query, mode=mode, number=size, instance=instance_url)
  final_tweets = []
  for tweet in tweets['tweets']:
    data = [tweet['link'], tweet['text'], tweet['date'], tweet['stats']['likes'], tweet['stats']['comments']]
    final_tweets.append(data)
    data = pd.DataFrame(final_tweets, columns = ['link', 'text', 'date', 'Likes', 'Comments'])
    # Convert DataFrame to JSON
    json_data = data.to_json(orient='records', lines=False)
    return json_data

@app.route('/store_tweets', methods=['POST'])
def store_tweets(): 
    # MongoDB setup
    mongo_url = 'mongodb://localhost:27017/'
    client = MongoClient(mongo_url)
    db = client['test_db']  # Your database name
    collection = db['collection_db']  # Your collection name
    
    # Call get_tweets function
    tweet_data = get_tweets("harcèlement","hashtag",5,"https://nitter.lucabased.xyz")

    if tweet_data is None:
        return jsonify({'error': 'No tweets found or an error occurred in get_tweets'}), 500

    
    # Convert tweet data (JSON string) to Python list of dictionaries
    tweet_list = pd.read_json(tweet_data, orient='records')
    
    # Insert tweets into MongoDB
    collection.insert_many(tweet_list.to_dict('records'))
    
    return jsonify({'message': 'Tweets stored successfully!'})


# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('french'))

def clean_text(text):
    # Lowercase conversion
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenization
    words = text.split()
    # Remove stop words and stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    # Join words back to string
    cleaned_text = ' '.join(words)
    return cleaned_text 

def detect_toxic(text):
  return Detoxify('original').predict(text)

# API route to get tweets for preprocessing
@app.route('/get_tweets', methods=['GET'])
def get_stored_tweets():
    # MongoDB connection (Inside Route Scope)
    mongo_url = 'mongodb://localhost:27017/'  # Update as needed
    client = MongoClient(mongo_url)
    db = client['test_db']  # Your database name
    collection = db['collection_db']  # Your collection name

    # Retrieve tweets from MongoDB
    tweets = list(collection.find({}, {'_id': 0, 'text': 1}))  # Exclude the MongoDB id field

    # Convert to DataFrame if you want to preprocess
    df_tweets = pd.DataFrame(tweets)
    
    df_tweets['cleaned']=df_tweets['text'].apply(clean_text)
    df_tweets["prediction"]=df_tweets['cleaned'].apply(detect_toxic)

    return jsonify(df_tweets.to_dict(orient='records'))




if __name__ == '__main__':
    app.run(debug=True)
