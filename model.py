import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import os
from scipy.sparse import coo_matrix

data=pd.read_csv('/content/marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv',sep='\t')

data.info()

data.head(1)

data.isnull().sum()

data.shape

req_data=data[['Uniq Id','Product Id','Product Rating','Product Reviews Count','Product Category','Product Brand','Product Name','Product Image Url','Product Description','Product Tags']]

req_data.isnull().sum()

req_data['Product Rating'].fillna(0,inplace=True)
req_data['Product Reviews Count'].fillna(0,inplace=True)

req_data['Product Category'].fillna("",inplace=True)
req_data['Product Brand'].fillna("",inplace=True)
req_data['Product Description'].fillna("",inplace=True)

req_data.isnull().sum()

req_data.columns

req_data.columns = ['id', 'prod_id', 'rating', 'reviews', 'category', 'brand', 'name', 'image_url', 'description', 'tags']


req_data.columns

req_data.head(1)


req_data['id']=req_data['id'].str.extract(r'(\d+)').astype(float)
req_data['prod_id']=req_data['prod_id'].str.extract(r'(\d+)').astype(float)

heatmap_data=req_data.pivot_table('id','rating')

plt.figure(figsize=(8,6))
sns.heatmap(heatmap_data,annot=True,fmt='g',cmap='coolwarm',cbar=True)
plt.title('Heatmap of User Ratings')
plt.xlabel('Rating')
plt.ylabel('User ID')
plt.show()


popular_items=req_data['prod_id'].value_counts().head(5)
popular_items.plot(kind='bar',color='red')
plt.title('Most Popular Items')
plt.xlabel('Product ID')
plt.ylabel('Number of Reviews')
plt.show()



req_data['rating'].value_counts().plot(kind='bar',color='red')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp=spacy.load('en_core_web_sm')
def clean_text(text):
  doc=nlp(text.lower())
  tags=[token.text for token in doc if token.text.isalnum() and token.text not in STOP_WORDS]
  return ' ,'.join(tags)

column_to_extract_tags=['category','brand','description']

for column in column_to_extract_tags:
  req_data[column]=req_data[column].apply(clean_text)





req_data['name'][0]



avg_rating=req_data.groupby(['name','reviews','brand','image_url'])['rating'].mean().reset_index()

top_rated_items=avg_rating.sort_values('rating',ascending=False)

rating_base_recommendation=top_rated_items.head(10)

rating_base_recommendation.head()

rating_base_recommendation['rating']=rating_base_recommendation['rating'].astype(int)
rating_base_recommendation['reviews']=rating_base_recommendation['reviews'].astype(int)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf_vectorizer=TfidfVectorizer(stop_words='english')
tfidf_matrix_content=tfidf_vectorizer.fit_transform(req_data['tags'])

cosine_similarity_content=cosine_similarity(tfidf_matrix_content,tfidf_matrix_content)

item_name='OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath'
item_idx=req_data[req_data['name']==item_name].index[0]
similar_items=list(enumerate(cosine_similarity_content[item_idx]))



similar_items=sorted(similar_items,key=lambda x:x[1],reverse=True)
top_similar_items=similar_items[1:10]
recommended_items_indices=[x[0] for x in top_similar_items]

req_data.iloc[recommended_items_indices][['name','reviews','brand']]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendation(req_data,item_name,top_n=10):
  if item_name not in req_data['name'].values:
    print(f"Item '{item_name}' not found in the dataset.")
    return pd.DataFrame()

  tfidf_vectorizer=TfidfVectorizer(stop_words='english')
  tfidf_matrix_content=tfidf_vectorizer.fit_transform(req_data['tags'])
  cosine_similarity_content=cosine_similarity(tfidf_matrix_content,tfidf_matrix_content)
  item_idx=req_data[req_data['name']==item_name].index[0]
  similar_items=list(enumerate(cosine_similarity_content[item_idx]))
  similar_items=sorted(similar_items,key=lambda x:x[1],reverse=True)
  top_similar_items=similar_items[1:top_n+1]
  recommended_items_indices=[x[0] for x in top_similar_items]
  recommended_items_details=req_data.iloc[recommended_items_indices][['name','reviews','brand']]
  return recommended_items_details




item_name='OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath'
contents=content_based_recommendation(req_data,item_name,top_n=8)
contents


user_item_matrix=req_data.pivot_table(index='id',columns='prod_id',values='rating',aggfunc='mean').fillna(0).astype(int)

user_similarity=cosine_similarity(user_item_matrix)

target_user_id=4
target_user_index=user_item_matrix.index.get_loc(target_user_id)

user_similarities=user_similarity[target_user_index]
similar_user_indices=user_similarities.argsort()[::-1][1:]

recommended_items=[]

for user_index in similar_user_indices:
  rated_by_similar_user=user_item_matrix.iloc[user_index]
  not_rated_by_target_user=(rated_by_similar_user==0) & (user_item_matrix.iloc[target_user_index]==0)

  recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:10])

recommended_items_details=req_data[req_data['prod_id'].isin(recommended_items)][[ 'name', 'reviews','brand', 'image_url','rating',]]

recommended_items_details.shape

def collaborative_based_recommendation(req_data, target_user_id, top_n=10):
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd

    user_item_matrix = req_data.pivot_table(index='id', columns='prod_id', values='rating', aggfunc='mean').fillna(0)

    user_similarity = cosine_similarity(user_item_matrix)

    target_user_index = user_item_matrix.index.get_loc(target_user_id)

    user_similarities = user_similarity[target_user_index]

    similar_user_indices = user_similarities.argsort()[::-1][1:]

    recommended_items = set()

    for user_index in similar_user_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user > 0) & (user_item_matrix.iloc[target_user_index] == 0)

        new_recommendations = user_item_matrix.columns[not_rated_by_target_user]

        for item in new_recommendations:
            if len(recommended_items) < top_n:
                recommended_items.add(item)
            else:
                break  
        if len(recommended_items) >= top_n:
            break


    recommended_items_details = req_data[req_data['prod_id'].isin(recommended_items)][['name', 'reviews', 'brand', 'image_url', 'rating']].drop_duplicates()

    return recommended_items_details.head(top_n)

target_user_id = 4
top_n = 5
collaborative_based_rec = collaborative_based_recommendation(req_data, target_user_id, top_n)

print(f"Top {top_n} recommendations for user {target_user_id}:")
collaborative_based_rec



def hybrid_recommendation(req_data, target_user_id, item_name, top_n=10):
    content_based_rec=content_based_recommendation(req_data,item_name,top_n)
    collaborative_based_rec=collaborative_based_recommendation(req_data,target_user_id,top_n)
    hybrid_rec=pd.concat([content_based_rec,collaborative_based_rec]).drop_duplicates()
    return hybrid_rec.head(10)


import pandas as pd


def get_recommendations(query, df):
    query = query.lower()

    results = df[df['name'].str.lower().str.contains(query, na=False)]

    recommendations = results['name'].unique().tolist()[:5]  
    return recommendations if recommendations else ["No recommendations found"]


