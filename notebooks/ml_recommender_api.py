"""
ML Recommender API Wrapper - Accepts JSON input and returns recommendations
"""

import os
import json
import base64
from typing import List, Dict, Optional
import pandas as pd
import google.generativeai as genai
from supabase import create_client, Client

# Import your existing ML code
from jewelry_ml_model import JewelryDataLoader, JewelryRecommender


class RecommendationAPI:
    """API wrapper for the jewelry recommendation system"""
    
    def __init__(self, 
                 csv_path: str = 'ready_for_ml_adjusted_if.csv',
                 supabase_url: str = None,
                 supabase_key: str = None,
                 gemini_api_key: str = None):
        
        # Initialize Supabase
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_KEY')
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize Gemini
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=self.gemini_api_key)
        self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Load and train ML model
        print("Loading ML model...")
        self.loader = JewelryDataLoader(csv_path)
        self.loader.load()
        
        self.recommender = JewelryRecommender(self.loader, n_clusters=8)
        self.recommender.fit()
        print("ML model ready!")
    
    def analyze_outfit_image(self, image_data: str) -> Dict:
        """
        Analyze outfit image using Gemini Vision API
        
        Args:
            image_data: Base64 encoded image string (with or without data URL prefix)
        
        Returns:
            Dict with 'colors' and 'styles' extracted from image
        """
        try:
            # Remove data URL prefix if present
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_data)
            
            # Create prompt for Gemini
            prompt = """Analyze this outfit image and extract:
1. Dominant colors (choose from: black, white, gray, silver, gold, pink, red, orange, yellow, green, blue, purple, brown, beige, neutral)
2. Style characteristics (choose from: classic, modern, romantic, bold, bohemian, minimalist, vintage, luxurious)

Return ONLY a JSON object with this exact format:
{
  "colors": ["color1", "color2", "color3"],
  "styles": ["style1", "style2"]
}

Be specific and choose 2-4 colors and 1-3 styles that best match the outfit."""
            
            # Call Gemini Vision API
            response = self.vision_model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": image_bytes}
            ])
            
            # Parse response
            response_text = response.text.strip()
            # Remove markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            result = json.loads(response_text)
            
            # Validate and normalize
            colors = [c.lower() for c in result.get('colors', [])]
            styles = [s.lower() for s in result.get('styles', [])]
            
            return {
                'colors': colors,
                'styles': styles
            }
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            # Return defaults if image analysis fails
            return {
                'colors': ['neutral'],
                'styles': ['classic']
            }
    
    def get_recommendations(self, request_data: Dict) -> Dict:
        """
        Main recommendation function
        
        Args:
            request_data: {
                'image': base64 image string (optional),
                'preferences': {
                    'styles': ['classic', 'modern'],
                    'budgetMin': 100,
                    'budgetMax': 5000,
                    'material': 'Gold' (optional)
                }
            }
        
        Returns:
            Dict with recommendations and metadata
        """
        
        # Step 1: Analyze image if provided
        image_analysis = {'colors': [], 'styles': []}
        if 'image' in request_data and request_data['image']:
            print("Analyzing outfit image...")
            image_analysis = self.analyze_outfit_image(request_data['image'])
            print(f"Image analysis: {image_analysis}")
        
        # Step 2: Combine with user preferences
        preferences = request_data.get('preferences', {})
        
        # Merge styles (user preferences + image analysis)
        user_styles = [s.lower() for s in preferences.get('styles', [])]
        all_styles = list(set(user_styles + image_analysis['styles']))
        
        # Use image colors (prioritize image colors over user input)
        preferred_colors = image_analysis['colors'] if image_analysis['colors'] else []
        
        # Price range
        min_price = preferences.get('budgetMin', 0)
        max_price = preferences.get('budgetMax', 380000)
        
        # Material filter (optional)
        material_pref = preferences.get('material', '').lower()
        
        print(f"Final query: styles={all_styles}, colors={preferred_colors}, price=${min_price}-${max_price}")
        
        # Step 3: Get ML recommendations
        ml_results = self.recommender.recommend(
            preferred_colors=preferred_colors,
            preferred_styles=all_styles,
            min_price=min_price,
            max_price=max_price,
            min_quantity=1,
            top_k=20,  # Get more candidates
            return_breakdown=False
        )
        
        if len(ml_results) == 0:
            return {
                'recommendations': [],
                'message': 'No items found matching your criteria',
                'debug': {
                    'styles': all_styles,
                    'colors': preferred_colors,
                    'price_range': [min_price, max_price]
                }
            }
        
        # Step 4: Cross-reference with Supabase inventory
        refs = ml_results['ref'].tolist()
        
        # Query Supabase for current stock
        response = self.supabase.table('available_jewelry').select(
            'id, ref, title, price, image_url, style_tags, color_tags, stock_qty'
        ).in_('ref', refs).execute()
        
        db_items = {item['ref']: item for item in response.data}
        
        # Step 5: Merge ML scores with DB data
        recommendations = []
        for _, row in ml_results.iterrows():
            ref = row['ref']
            if ref in db_items:
                db_item = db_items[ref]
                recommendations.append({
                    'id': db_item['id'],
                    'ref': ref,
                    'name': db_item['title'],
                    'price': float(row['price']),
                    'image_url': db_item['image_url'],
                    'style_tags': db_item.get('style_tags', []),
                    'color_tags': db_item.get('color_tags', []),
                    'material': material_pref if material_pref else 'Mixed',
                    'match_score': int(row['total_score']),
                    'style_score': float(row['style_score']),
                    'color_score': float(row['color_score']),
                    'cluster_bonus': float(row['cluster_bonus']),
                    'cluster_style': row['cluster_style'],
                    'stock_qty': db_item['stock_qty']
                })
        
        # Sort by match_score and limit to top 5
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        top_recommendations = recommendations[:5]
        
        return {
            'recommendations': top_recommendations,
            'total_found': len(recommendations),
            'image_analysis': image_analysis,
            'query_params': {
                'styles': all_styles,
                'colors': preferred_colors,
                'price_range': [min_price, max_price]
            }
        }
    
    def log_recommendation(self, session_id: str, recommendations: List[Dict]):
        """Log recommendations to database"""
        try:
            logs = []
            for rank, item in enumerate(recommendations, 1):
                logs.append({
                    'session_id': session_id,
                    'item_id': item['id'],
                    'rank': rank,
                    'total_score': item['match_score'],
                    'style_score': item['style_score'],
                    'color_score': item['color_score'],
                    'cluster_bonus': item['cluster_bonus']
                })
            
            self.supabase.table('recommendation_log').insert(logs).execute()
            print(f"Logged {len(logs)} recommendations")
        except Exception as e:
            print(f"Error logging recommendations: {e}")


# Flask API endpoint (for testing)
if __name__ == "__main__":
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    # Initialize API
    api = RecommendationAPI()
    
    @app.route('/recommend', methods=['POST'])
    def recommend():
        try:
            data = request.json
            results = api.get_recommendations(data)
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy'})
    
    app.run(host='0.0.0.0', port=5000, debug=True)