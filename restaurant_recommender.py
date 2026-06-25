import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# STEP 1: DATA PREPROCESSING (HANDLING MISSING VALUES)
# =====================================================================
print("--- Starting Step 1: Data Preprocessing ---")

# Load your uploaded CSV file
df = pd.read_csv("Dataset .csv")

# Handle missing values in the 'Cuisines' column by filling with a placeholder
df['Cuisines'] = df['Cuisines'].fillna('Unknown')

print(f"Dataset loaded successfully with {len(df)} rows.")


# =====================================================================
# STEP 2: CATEGORICAL ENCODING & FEATURE ENGINEERING
# =====================================================================
print("\n--- Starting Step 2: Feature Engineering & Encoding ---")

# 1. Multi-hot encode Cuisines (Handles comma-separated multi-values per row)
cuisine_features = df['Cuisines'].str.get_dummies(sep=', ')

# 2. One-hot encode Price Range category
price_features = pd.get_dummies(df['Price range'], prefix='PriceRange', dtype=int)

# 3. Scale Aggregate Rating numerical values between 0 and 1
scaler = MinMaxScaler()
df['ScaledRating'] = scaler.fit_transform(df[['Aggregate rating']])
rating_features = df[['ScaledRating']]

# 4. Merge all processed columns into a unified numerical Feature Matrix
features_df = pd.concat([cuisine_features, price_features, rating_features], axis=1)

print(f"Feature matrix engineered successfully. Shape: {features_df.shape}")


# =====================================================================
# STEP 3: IMPLEMENT CONTENT-BASED FILTERING ENGINE
# =====================================================================
print("\n--- Step 3: Recommendation Engine Loaded ---")

def recommend_restaurants(user_prefs, original_df, processed_features, top_n=5):
    """
    Applies local filtering and Cosine Similarity to recommend restaurants.
    """
    # 1. Contextual Location Filtering (Filter by City if provided)
    city = user_prefs.get('city')
    if city:
        city_mask = original_df['City'].str.lower() == city.lower()
        filtered_df = original_df[city_mask].copy()
        filtered_features = processed_features[city_mask].copy()
    else:
        filtered_df = original_df.copy()
        filtered_features = processed_features.copy()
        
    if filtered_df.empty:
        return f"No restaurants found in the city: {city}"
        
    # 2. Construct User Vector matching the engineered feature columns
    user_vector = np.zeros(processed_features.shape[1])
    feature_columns = list(processed_features.columns)
    
    # Map requested cuisine styles
    for cuisine in user_prefs.get('cuisines', []):
        if cuisine in feature_columns:
            user_vector[feature_columns.index(cuisine)] = 1.0
            
    # Map requested price range tier
    pref_price = user_prefs.get('price_range')
    if pref_price is not None:
        price_col = f'PriceRange_{pref_price}'
        if price_col in feature_columns:
            user_vector[feature_columns.index(price_col)] = 1.0
            
    # Map user rating importance weight
    if 'ScaledRating' in feature_columns:
        user_vector[feature_columns.index('ScaledRating')] = user_prefs.get('rating_importance', 0.5)
        
    # 3. Compute Cosine Similarity between user profile vector and data rows
    user_vector = user_vector.reshape(1, -1)
    similarity_scores = cosine_similarity(filtered_features, user_vector).flatten()
    
    # Store scores back into the filtered view
    filtered_df['Match Score'] = similarity_scores
    
    # 4. Sort Candidates dynamically: Similarity Match -> Aggregate Rating -> Votes
    recommendations = filtered_df.sort_values(
        by=['Match Score', 'Aggregate rating', 'Votes'], 
        ascending=False
    )
    
    return recommendations.head(top_n)


# =====================================================================
# STEP 4: SYSTEM TESTING & EVALUATION
# =====================================================================
print("\n--- Starting Step 4: System Testing & Evaluation ---")

# Defining sample user criteria for assessment
sample_preferences = {
    'city': 'New Delhi',
    'cuisines': ['Italian', 'Continental'],
    'price_range': 3,
    'rating_importance': 1.0
}

# Generate top 3 recommendations
results = recommend_restaurants(sample_preferences, df, features_df, top_n=3)

# Print out your final proof of execution results
print("\n======================= TEST CASE RESULTS =======================")
print(results[['Restaurant Name', 'City', 'Cuisines', 'Price range', 'Aggregate rating', 'Match Score']])
print("=================================================================\n")
print("Execution Complete! Take a screenshot of this output for your report.")
# =====================================================================
# STEP 5: VISUAL REPRESENTATION (FOR VIDEO DEMO)
# =====================================================================
print("\n--- Generating Visual Charts for Project Demo ---")
import matplotlib.pyplot as plt
import seaborn as sns

# चार्ट्स का स्टाइल सेट करें
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# 1. विज़ुअल: डेटासेट में सबसे पॉपुलर टॉप 7 कुजीन्स
plt.figure()
top_cuisines = cuisine_features.sum().sort_values(ascending=False).head(7)
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, hue=top_cuisines.index, palette="viridis", legend=False)
plt.title("Top 7 Most Popular Cuisines in the Dataset", fontsize=14, fontweight='bold')
plt.xlabel("Number of Restaurants")
plt.ylabel("Cuisine Type")
plt.tight_layout()
plt.savefig("top_cuisines.png", dpi=300) # इमेज सेव होगी
plt.show()

# 2. विज़ुअल: आपके टेस्ट केस के टॉप रिकमेंडेड रेस्टोरेंट्स का मैच स्कोर
if isinstance(results, pd.DataFrame) and not results.empty:
    plt.figure()
    sns.barplot(x="Match Score", y="Restaurant Name", data=results, hue="Restaurant Name", palette="magma", legend=False)
    plt.xlim(0, 1.1) # स्कोर 0 से 1 के बीच होता है
    plt.title("Top Recommended Restaurants & Their Match Scores", fontsize=14, fontweight='bold')
    plt.xlabel("Cosine Similarity (Match Score)")
    plt.ylabel("Restaurant Name")
    
    # बार के ऊपर स्कोर का टेक्स्ट लिखने के लिए
    for index, value in enumerate(results['Match Score']):
        plt.text(value + 0.02, index, f"{value:.4f}", va='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("recommendation_results.png", dpi=300) # इमेज सेव होगी
    plt.show()

print("Visualizations generated successfully! Images saved as 'top_cuisines.png' and 'recommendation_results.png'.")