from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# ======= KHỞI TẠO FLASK APP =======
app = Flask(__name__)
CORS(app)

# ======= KẾT NỐI MONGODB =======
# uri = "mongodb+srv://lhdhuy124:O5ZFsMYwjHJMK4Dv@cluster0.ldwlz.mongodb.net/beauty-box"
import os
uri = os.environ.get("MONGO_URI")

client = MongoClient(uri)
db = client["beauty-box"]
products_col = db["products"]
categories_col = db["categorys"]  # Giữ nguyên tên collection
brands_col = db["brands"]
reviews_col = db["reviews"]

# ======= LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU =======
print("📦 Đang tải dữ liệu MongoDB...")
products = list(products_col.find({"deleted": False}, {"_id": 1, "title": 1, "category_id": 1, "brand_id": 1, "thumbnail": 1, "price": 1, "discountPercentage": 1, "slug": 1}))
df_products = pd.DataFrame(products)
df_products['_id'] = df_products['_id'].astype(str)
df_products['category_id'] = df_products['category_id'].astype(str)
df_products['brand_id'] = df_products['brand_id'].astype(str)

categories = list(categories_col.find({}, {"_id": 1, "title": 1}))
df_categories = pd.DataFrame(categories)
df_categories['_id'] = df_categories['_id'].astype(str)
df_categories.rename(columns={'_id': 'category_id', 'title': 'category_title'}, inplace=True)

brands = list(brands_col.find({}, {"_id": 1, "name": 1}))
df_brands = pd.DataFrame(brands)
df_brands['_id'] = df_brands['_id'].astype(str)
df_brands.rename(columns={'_id': 'brand_id', 'name': 'brand_title'}, inplace=True)

df_products.rename(columns={'_id': 'product_id'}, inplace=True)

# Merge bảng category và brand vào product
df_merged = pd.merge(df_products, df_categories, on='category_id', how='left')
df_merged = pd.merge(df_merged, df_brands, on='brand_id', how='left')

# ======= TF-IDF TRÊN CATEGORY (CONTENT-BASED) =======
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df_merged['category_title'])

content_sim_df = pd.DataFrame(
    cosine_similarity(tfidf_matrix),
    index=df_merged['title'],
    columns=df_merged['title']
)

# ======= COLLABORATIVE FILTERING =======
reviews = list(reviews_col.find({"public": True}, {"user_id": 1, "product_id": 1, "rating": 1}))
df_reviews = pd.DataFrame(reviews)
df_reviews['product_id'] = df_reviews['product_id'].astype(str)

df_products_all = pd.DataFrame(products)
df_products_all['_id'] = df_products_all['_id'].astype(str)
df_reviews = pd.merge(df_reviews, df_products_all[['_id', 'title']], left_on='product_id', right_on='_id', how='left')

user_item_matrix = df_reviews.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

collab_sim_df = pd.DataFrame(
    cosine_similarity(user_item_matrix.T),
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

# ======= HYBRID SIMILARITY =======
common_titles = content_sim_df.index.intersection(collab_sim_df.index)
content_sim_df = content_sim_df.loc[common_titles, common_titles]
collab_sim_df = collab_sim_df.loc[common_titles, common_titles]

alpha = 0.5
hybrid_sim_df = alpha * content_sim_df + (1 - alpha) * collab_sim_df

# ======= TF-IDF TRÊN TIÊU ĐỀ (SEARCH) =======
title_vectorizer = TfidfVectorizer(stop_words='english')
title_tfidf = title_vectorizer.fit_transform(df_merged['title'])

def search_title_by_similarity(input_title, top_n=1):
    input_vec = title_vectorizer.transform([input_title])
    cosine_sim = linear_kernel(input_vec, title_tfidf).flatten()
    top_indices = cosine_sim.argsort()[::-1][:top_n]
    return df_merged.iloc[top_indices]['title'].tolist()

# ======= HÀM GỢI Ý =======
def recommend_hybrid(title, top_n=5):
    if title not in hybrid_sim_df.index:
        return None

    scores = hybrid_sim_df[title].sort_values(ascending=False).drop(title).head(top_n)
    result = pd.DataFrame({
        'title': scores.index,
        'similarity': scores.values
    })

    # merge thêm thumbnail
    result = pd.merge(result, df_merged[['product_id', 'title', 'category_title', 'thumbnail', 'slug', 'price', 'discountPercentage']], on='title', how='left')

    ratings_stats = df_reviews.groupby('title')['rating'].agg(['mean', 'count']).reset_index()
    ratings_stats.rename(columns={'mean': 'avg_rating', 'count': 'count_rating'}, inplace=True)

    result = pd.merge(result, ratings_stats, on='title', how='left')

    return result[['product_id', 'title', 'category_title', 'thumbnail', 'avg_rating', 'count_rating', 'similarity', 'slug', 'price', 'discountPercentage']]

def search_and_recommend_by_title(input_title, top_n=5):
    matched_titles = search_title_by_similarity(input_title, top_n=1)
    if not matched_titles:
        return None

    best_match = matched_titles[0]
    return recommend_hybrid(best_match, top_n=top_n)

def recommend_hybrid_by_product_id(product_id, top_n=5):
    if product_id not in df_products['product_id'].values:
        return None

    product_title = df_products[df_products['product_id'] == product_id]['title'].values[0]
    if product_title not in hybrid_sim_df.index:
        return None

    scores = hybrid_sim_df[product_title].sort_values(ascending=False).drop(product_title).head(top_n)
    result = pd.DataFrame({
        'title': scores.index,
        'similarity': scores.values
    })

    result = pd.merge(result, df_merged[['product_id', 'title', 'category_title', 'brand_title', 'price', 'discountPercentage', 'slug', 'thumbnail']], on='title', how='left')

    ratings_stats = df_reviews.groupby('title')['rating'].agg(['mean', 'count']).reset_index()
    ratings_stats.rename(columns={'mean': 'avg_rating', 'count': 'count_rating'}, inplace=True)
    result = pd.merge(result, ratings_stats, on='title', how='left')

    return result[['product_id', 'title', 'category_title', 'brand_title', 'price', 'discountPercentage', 'slug', 'thumbnail', 'avg_rating', 'similarity', 'count_rating']]

# ======= API FLASK =======
@app.route('/recommend/by-title', methods=['GET'])
def api_recommend_by_title():
    input_title = request.args.get('title')
    top_n = int(request.args.get('top_n', 5))

    if not input_title:
        return jsonify({'error': 'Missing "title" parameter'}), 400

    result = search_and_recommend_by_title(input_title, top_n)
    if result is None:
        return jsonify({'message': 'Không tìm thấy sản phẩm hoặc gợi ý'}), 404

    return jsonify(result.to_dict(orient='records'))

@app.route('/recommend/by-id', methods=['GET'])
def api_recommend_by_id():
    product_id = request.args.get('product_id')
    top_n_str = request.args.get('top_n', '5')
    try:
        top_n = int(top_n_str.strip('"'))
    except (ValueError, AttributeError):
        return jsonify({'error': 'Invalid "top_n" parameter'}), 400

    if not product_id:
        return jsonify({'error': 'Missing "product_id" parameter'}), 400

    result = recommend_hybrid_by_product_id(product_id, top_n)
    if result is None:
        return jsonify({'message': 'Không tìm thấy sản phẩm hoặc gợi ý'}), 404

    return jsonify(result.to_dict(orient='records'))

# ======= RUN SERVER =======
# if __name__ == '__main__':
#     print("🚀 Flask server đang chạy tại http://127.0.0.1:3030")
#     app.run(debug=True, port=3030)
import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
