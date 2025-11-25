import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List

class MLBasedRecommender:
    """
    Machine Learning based jewelry recommender using TF-IDF and cosine similarity
    Implements content-based filtering with theme mapping
    """

    def __init__(self, data_path: str):
        """
        Initialize the ML recommender

        Args:
            data_path: Path to the jewelry CSV data
        """
        self.df_items = pd.read_csv(data_path)
        self.MAX_STOCK = 50
        self.LOW_STOCK_THRESHOLD = 5
        self.STOCK_ADJUSTMENT_FACTOR = 0.9

        # Theme weights for different occasions
        self.THEME_WEIGHTS = {
            "Formal Evening Gala": {
                'necklaces': 1.0, 'diamond': 0.9, 'white gold': 0.7, 'price_norm': 0.8
            },
            "Casual Beach Day": {
                'bracelets': 0.9, 'aquamarines': 1.0, 'yellow gold': 0.5, 'price_norm': 0.1
            },
            "Business Professional": {
                'earrings': 0.8, 'platinum': 0.9, 'price_norm': 0.5
            },
            "Romantic Date Night": {
                'rings': 0.7, 'pink sapphire': 1.0, 'pink gold': 1.0, 'price_norm': 0.6
            },
            "Outdoor Adventure": {
                'bracelets': 0.7, 'carnelian': 0.8, 'yellow gold': 0.3, 'price_norm': 0.1
            },
            "Default Theme": {'price_norm': 0.1}
        }

        self._preprocess_data()
        self._setup_tfidf()

    def _preprocess_data(self):
        """Consolidate features and normalize prices"""
        # Consolidate singular/plural features
        if 'diamond' in self.df_items.columns and 'diamonds' in self.df_items.columns:
            self.df_items['diamond'] = self.df_items['diamond'] | self.df_items['diamonds']

        # Normalize price
        scaler = MinMaxScaler()
        self.df_items['price_norm'] = scaler.fit_transform(self.df_items[['price']])

        # Define content features (adjust based on available columns)
        self.CONTENT_FEATURES = ['price_norm']

        # Add available material and category features
        potential_features = ['diamond', 'amethyst', 'sapphire', 'pink sapphire',
                            'yellow gold', 'white gold', 'platinum', 'pink gold',
                            'rings', 'earrings', 'necklaces', 'bracelets']

        for feature in potential_features:
            if feature in self.df_items.columns:
                self.CONTENT_FEATURES.append(feature)

        print(f"Total features used by the model: {len(self.CONTENT_FEATURES)}")

    def _setup_tfidf(self):
        """Setup TF-IDF vectorizer for text preferences"""
        corpus = self.df_items['title'].tolist()
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w+\b')
        self.tfidf_vectorizer.fit(corpus)
        self.vocab = self.tfidf_vectorizer.get_feature_names_out()

    def _map_nlp_to_features(self, word: str, score: float, U_nlp_vector: pd.Series):
        """Map NLP words to jewelry features"""
        if 'blue' in word and 'sapphire' in self.CONTENT_FEATURES:
            U_nlp_vector['sapphire'] = max(U_nlp_vector.get('sapphire', 0), score)
        if ('yellow' in word or 'metal' in word) and 'yellow gold' in self.CONTENT_FEATURES:
            U_nlp_vector['yellow gold'] = max(U_nlp_vector.get('yellow gold', 0), score)
        if ('sparkle' in word or 'diamond' in word) and 'diamond' in self.CONTENT_FEATURES:
            U_nlp_vector['diamond'] = max(U_nlp_vector.get('diamond', 0), score)

    def recommend(self,
                 theme: str,
                 preferences_text: str,
                 item_type_focus: str,
                 price_min: float,
                 price_max: float,
                 top_k: int = 5,
                 price_tolerance: float = 0.05) -> pd.DataFrame:
        """
        Generate recommendations based on user context

        Args:
            theme: Event theme (e.g., "Romantic Date Night")
            preferences_text: User's text preferences
            item_type_focus: Category focus (rings, earrings, etc.)
            price_min: Minimum price
            price_max: Maximum price
            top_k: Number of recommendations
            price_tolerance: Price range tolerance (default 5%)

        Returns:
            DataFrame with top recommendations
        """
        # Build combined user vector
        U_nlp_vector = pd.Series(0.0, index=self.CONTENT_FEATURES)

        # Process text preferences with TF-IDF
        user_tfidf_sparse = self.tfidf_vectorizer.transform([preferences_text])
        user_tfidf_vector = user_tfidf_sparse.toarray()[0]

        for word, score in zip(self.vocab, user_tfidf_vector):
            if score > 0:
                self._map_nlp_to_features(word, score, U_nlp_vector)

        # Get theme vector
        theme_data = self.THEME_WEIGHTS.get(theme, self.THEME_WEIGHTS['Default Theme'])
        U_theme_vector = pd.Series(theme_data).reindex(self.CONTENT_FEATURES, fill_value=0)

        # Type focus vector
        U_type_vector = pd.Series(0.0, index=self.CONTENT_FEATURES)
        if item_type_focus in self.CONTENT_FEATURES:
            U_type_vector[item_type_focus] = 1.0

        # Combine vectors with weights
        ALPHA, BETA, GAMMA = 0.6, 0.3, 0.1
        U_combined = (ALPHA * U_theme_vector + BETA * U_nlp_vector + GAMMA * U_type_vector)
        U = U_combined.values.reshape(1, -1)

        # Filter by price and stock
        min_p = price_min * (1 - price_tolerance)
        max_p = price_max * (1 + price_tolerance)

        df_filtered = self.df_items[
            (self.df_items['quantity'] > 0) &
            (self.df_items['price'] >= min_p) &
            (self.df_items['price'] <= max_p)
        ].copy()

        if df_filtered.empty:
            print("\nNo items matched the price and stock constraints.")
            return pd.DataFrame()

        # Calculate similarity scores
        I_filtered = df_filtered[self.CONTENT_FEATURES].values
        similarity_scores = cosine_similarity(U, I_filtered)[0]
        df_filtered['similarity_score'] = similarity_scores

        # Apply stock adjustment
        is_low_stock = df_filtered['quantity'] <= self.LOW_STOCK_THRESHOLD
        df_filtered['stock_adjustment'] = np.where(is_low_stock, self.STOCK_ADJUSTMENT_FACTOR, 1.0)

        # Calculate final score
        df_filtered['final_score'] = df_filtered['similarity_score'] * df_filtered['stock_adjustment']

        # Return top K recommendations
        recommendations = df_filtered.sort_values(
            by='final_score',
            ascending=False
        ).head(top_k)[['ref', 'title', 'price', 'quantity', 'final_score']]

        return recommendations
