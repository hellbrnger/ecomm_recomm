from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load Data
data_path = "data/marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv"
data = pd.read_csv(data_path, sep="\t")

# Preprocessing
data = data[['Uniq Id', 'Product Id', 'Product Rating', 'Product Reviews Count', 'Product Name', 'Product Description',
             'Product Image Url']]
data.columns = ['id', 'prod_id', 'rating', 'reviews', 'name', 'description', 'image']
data.fillna("", inplace=True)

# TF-IDF for Content-Based Recommendation
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Recommendation Function
def content_based_recommendation(item_name, top_n=10):
    if item_name not in data['name'].values:
        return []

    item_idx = data[data['name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_sim[item_idx]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    recommended_indices = [x[0] for x in similar_items]

    return data.iloc[recommended_indices][['name', 'reviews', 'image']].to_dict(orient='records')


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        item_name = request.form.get("item_name")
        recommendations = content_based_recommendation(item_name)

    return render_template("index.html", recommendations=recommendations)


@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    query = request.args.get("q", "")
    matches = data[data['name'].str.contains(query, case=False, na=False)]['name'].tolist()
    return jsonify(matches)


if __name__ == "__main__":
    app.run(debug=True)
