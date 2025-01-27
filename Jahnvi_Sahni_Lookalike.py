import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data
customers = pd.read_csv('data/Customers.csv')
transactions = pd.read_csv('data/Transactions.csv')
products = pd.read_csv('data/Products.csv')

# Feature Engineering
def create_features(customers, transactions, products):
    # Merge data
    data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')
    
    # Aggregate features
    features = data.groupby('CustomerID').agg({
        'TotalValue': ['mean', 'sum'],
        'ProductID': 'count',
        'Category': lambda x: x.mode()[0]
    }).reset_index()
    
    features.columns = ['CustomerID', 'AvgTransactionValue', 'TotalSpend', 'PurchaseCount', 'FavoriteCategory']
    return features

features = create_features(customers, transactions, products)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), ['AvgTransactionValue', 'TotalSpend', 'PurchaseCount']),
        ('cat', OneHotEncoder(), ['FavoriteCategory'])
    ])

# Transform features
X = preprocessor.fit_transform(features)

# Calculate similarity
similarity_matrix = cosine_similarity(X)

# Get top 3 lookalikes for each customer
lookalikes = {}
for idx, customer_id in enumerate(features['CustomerID']):
    similar_indices = similarity_matrix[idx].argsort()[-4:-1][::-1]  # Exclude self
    similar_customers = [(features['CustomerID'][i], similarity_matrix[idx][i]) for i in similar_indices]
    lookalikes[customer_id] = similar_customers

# Create Lookalike.csv
lookalike_df = pd.DataFrame({
    'CustomerID': list(lookalikes.keys()),
    'Lookalikes': [str(v) for v in lookalikes.values()]
})

lookalike_df.to_csv('Jahnvi_sahni_Lookalike.csv', index=False)

# Output for first 20 customers
print(lookalike_df.head(20))