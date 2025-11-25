import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import json

class JewelryRecommendationSystem:
    """
    Agentic AI + E-Commerce Database Jewelry Recommendation System
    Based on the project proposal for styling and sales uplift
    """

    def __init__(self, jewelry_data_path: str = None):
        """
        Initialize the recommendation system

        Args:
            jewelry_data_path: Path to CSV file with jewelry inventory
        """
        self.jewelry_df = None
        self.recommendation_log = []

        # Define style vocabularies for consistent tagging
        self.style_tags = ['elegant', 'minimalist', 'boho', 'classic',
                          'modern', 'vintage', 'casual', 'formal']

        # Material categories from your data columns
        self.materials = ['amazonite', 'amethyst', 'aquamarines', 'aventurine',
                         'black ceramic', 'carnelian', 'ceramic', 'chrysoprase',
                         'citrine', 'coral', 'diamond', 'emeralds', 'garnets',
                         'lapis lazuli', 'malachite', 'mother_of_pearl', 'obsidians',
                         'onyx', 'pearl', 'peridots', 'pink gold', 'pink sapphire',
                         'platinum', 'rubies', 'sapphire', 'spessartite garnet',
                         'spinels', 'tsavorite garnet', 'white gold', 'yellow gold']

        # Category columns from your data
        self.categories = ['bracelets', 'earrings', 'necklaces', 'rings']

        if jewelry_data_path:
            self.load_data(jewelry_data_path)

    def load_data(self, data_path: str):
        """Load jewelry inventory data from CSV"""
        self.jewelry_df = pd.read_csv(data_path)
        print(f"Loaded {len(self.jewelry_df)} jewelry items")

        # Preprocess: ensure quantity is numeric
        self.jewelry_df['quantity'] = pd.to_numeric(self.jewelry_df['quantity'], errors='coerce').fillna(0)
        self.jewelry_df['price'] = pd.to_numeric(self.jewelry_df['price'], errors='coerce')

    def create_sample_data(self, n_items: int = 100):
        """
        Create sample jewelry data matching your schema
        Useful for testing when you don't have the full dataset yet
        """
        np.random.seed(42)

        data = {
            'ref': [f'JEWEL{str(i).zfill(4)}' for i in range(n_items)],
            'title': [self._generate_title() for _ in range(n_items)],
            'price': np.random.randint(50, 2000, n_items),
            'image': [f'https://example.com/jewelry/{i}.jpg' for i in range(n_items)],
            'quantity': np.random.randint(0, 50, n_items),
        }

        # Add category columns (one-hot encoded)
        for cat in self.categories:
            data[cat] = np.random.choice([0, 1], n_items, p=[0.75, 0.25])

        # Add material columns (one-hot encoded)
        for mat in self.materials:
            data[mat] = np.random.choice([0, 1], n_items, p=[0.9, 0.1])

        self.jewelry_df = pd.DataFrame(data)
        print(f"Created {n_items} sample jewelry items")

    def _generate_title(self):
        """Generate realistic jewelry titles"""
        adj = np.random.choice(['Classic', 'Elegant', 'Modern', 'Vintage', 'Delicate', 'Bold'])
        material = np.random.choice(['Gold', 'Silver', 'Diamond', 'Pearl', 'Gemstone'])
        item = np.random.choice(['Ring', 'Necklace', 'Earrings', 'Bracelet'])
        return f"{adj} {material} {item}"

    def extract_item_features(self, item_row):
        """
        Extract features from a jewelry item row
        Returns: dict with category, materials, and inferred styles
        """
        features = {
            'category': None,
            'materials': [],
            'styles': []
        }

        # Extract category
        for cat in self.categories:
            if cat in item_row and item_row[cat] == 1:
                features['category'] = cat
                break

        # Extract materials
        for mat in self.materials:
            if mat in item_row and item_row[mat] == 1:
                features['materials'].append(mat)

        # Infer styles based on materials and category
        features['styles'] = self._infer_styles(features)

        return features

    def _infer_styles(self, features: Dict) -> List[str]:
        """
        Infer style tags from materials and category
        This is a rule-based heuristic; you can improve with ML later
        """
        styles = []
        materials = features.get('materials', [])

        # Material-based style inference
        if 'diamond' in materials or 'platinum' in materials:
            styles.extend(['elegant', 'formal', 'classic'])
        if 'pearl' in materials:
            styles.extend(['elegant', 'vintage'])
        if 'yellow gold' in materials or 'white gold' in materials:
            styles.extend(['classic', 'modern'])
        if 'pink gold' in materials:
            styles.extend(['modern', 'minimalist'])
        if any(stone in materials for stone in ['turquoise', 'carnelian', 'malachite']):
            styles.append('boho')

        # Category-based style inference
        category = features.get('category')
        if category == 'rings':
            styles.append('formal')
        elif category == 'bracelets':
            styles.extend(['casual', 'boho'])
        elif category == 'necklaces':
            styles.extend(['elegant', 'classic'])

        # Remove duplicates and ensure we have at least 2-3 styles
        styles = list(set(styles))
        if not styles:
            styles = ['classic', 'modern']

        return styles[:5]

    def recommend_jewelry(self,
                         budget_min: float,
                         budget_max: float,
                         style_preferences: List[str] = None,
                         event: str = None,
                         category: str = None,
                         material_preferences: List[str] = None,
                         top_k: int = 6,
                         session_id: str = None) -> List[Dict]:
        """
        Main recommendation function implementing the scoring logic from your proposal

        Args:
            budget_min: Minimum price
            budget_max: Maximum price
            style_preferences: List of style tags (e.g., ['minimalist', 'modern'])
            event: Event type (e.g., 'wedding', 'casual')
            category: Jewelry category (e.g., 'rings', 'necklaces')
            material_preferences: List of preferred materials
            top_k: Number of recommendations to return
            session_id: Session identifier for logging

        Returns:
            List of recommended items with scores and explanations
        """
        if self.jewelry_df is None:
            raise ValueError("No data loaded. Call load_data() or create_sample_data() first.")

        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # STEP 1: Apply hard filters
        filtered_df = self.jewelry_df.copy()

        # Filter 1: Must be in stock
        filtered_df = filtered_df[filtered_df['quantity'] > 0]

        # Filter 2: Must be within budget
        filtered_df = filtered_df[
            (filtered_df['price'] >= budget_min) &
            (filtered_df['price'] <= budget_max)
        ]

        # Filter 3: Category filter (if specified)
        if category:
            if category in self.categories:
                filtered_df = filtered_df[filtered_df[category] == 1]

        # Filter 4: Material filter (if specified)
        if material_preferences:
            material_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
            for material in material_preferences:
                if material in filtered_df.columns:
                    material_mask |= (filtered_df[material] == 1)
            filtered_df = filtered_df[material_mask]

        if len(filtered_df) == 0:
            print("No items match your filters. Try broadening your criteria.")
            return []

        # STEP 2: Score and rank items
        recommendations = []
        budget_mid = (budget_min + budget_max) / 2
        budget_range = budget_max - budget_min

        for idx, row in filtered_df.iterrows():
            # Extract item features
            item_features = self.extract_item_features(row)
            item_styles = item_features['styles']

            # Calculate style fit (overlap count)
            style_overlap = 0
            if style_preferences:
                style_overlap = len(set(style_preferences) & set(item_styles))

            # Calculate price fit (favor items near budget midpoint)
            price_diff = abs(row['price'] - budget_mid)
            price_fit = 1 - (price_diff / budget_range) if budget_range > 0 else 1
            price_fit = max(0, price_fit)

            # Final score: overlap + 0.5 × price_fit (from your proposal)
            final_score = style_overlap + (0.5 * price_fit)

            # Build recommendation object
            recommendation = {
                'ref': row['ref'],
                'title': row['title'],
                'price': float(row['price']),
                'image': row['image'],
                'quantity': int(row['quantity']),
                'category': item_features['category'],
                'materials': item_features['materials'],
                'styles': item_styles,
                'score': final_score,
                'style_overlap': style_overlap,
                'price_fit': price_fit,
                'explanation': self._generate_explanation(style_overlap, price_fit, item_features, style_preferences)
            }

            recommendations.append(recommendation)

        # STEP 3: Sort by score and return top K
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        top_recommendations = recommendations[:top_k]

        # STEP 4: Log recommendations
        self._log_recommendations(session_id, top_recommendations, {
            'budget_min': budget_min,
            'budget_max': budget_max,
            'style_preferences': style_preferences,
            'event': event,
            'category': category,
            'material_preferences': material_preferences
        })

        return top_recommendations

    def _generate_explanation(self, style_overlap, price_fit, features, user_styles):
        """Generate human-readable explanation for why item was recommended"""
        explanations = []

        if style_overlap > 0:
            matching_styles = set(features['styles']) & set(user_styles or [])
            if matching_styles:
                explanations.append(f"Matches your {', '.join(list(matching_styles)[:2])} style")

        if price_fit > 0.7:
            explanations.append("Great value within your budget")

        if not explanations:
            explanations.append("In stock and within budget")

        return "; ".join(explanations)

    def _log_recommendations(self, session_id, recommendations, user_inputs):
        """Log recommendations for analytics (simulating MongoDB storage)"""
        log_entry = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'user_inputs': user_inputs,
            'recommendations': [
                {
                    'ref': rec['ref'],
                    'rank': idx + 1,
                    'score': rec['score'],
                    'price': rec['price']
                }
                for idx, rec in enumerate(recommendations)
            ],
            'num_results': len(recommendations)
        }

        self.recommendation_log.append(log_entry)

    def get_recommendation_log(self) -> List[Dict]:
        """Retrieve recommendation logs for analytics"""
        return self.recommendation_log

    def analyze_trends(self) -> Dict:
        """
        Analyze recommendation logs to identify trends
        Useful for merchandising team use case
        """
        if not self.recommendation_log:
            return {"message": "No logs available yet"}

        all_inputs = [log['user_inputs'] for log in self.recommendation_log]

        # Count most requested styles
        style_counts = {}
        for inputs in all_inputs:
            styles = inputs.get('style_preferences', [])
            for style in styles:
                style_counts[style] = style_counts.get(style, 0) + 1

        # Count most requested categories
        category_counts = {}
        for inputs in all_inputs:
            cat = inputs.get('category')
            if cat:
                category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            'total_sessions': len(self.recommendation_log),
            'popular_styles': sorted(style_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'popular_categories': sorted(category_counts.items(), key=lambda x: x[1], reverse=True),
            'avg_budget': np.mean([inp.get('budget_max', 0) for inp in all_inputs])
        }
