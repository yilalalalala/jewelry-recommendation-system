"""
Jewelry Recommendation API - FastAPI with K-Modes ML + SQLAlchemy ORM

Combines:
- K-Modes clustering and scoring logic (from jewelry_ml_model.py)
- SQLAlchemy ORM for database access
- Gemini Vision for image analysis

Author: Cao & Ngo
Course: CSCI-GA.2433-001 Database Systems
Project: Part 4 - End-to-End Solution with ORM + ML
"""

import os
import json
import base64
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import Counter

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session

from models import (
    Base, Category, Material, Cluster, JewelryItem,
    ItemMaterial, Inventory, UserProfile, OutfitSession,
    RecommendationLog, get_engine
)

# Gemini for image analysis
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Image analysis disabled.")

# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres.onvusagrrakeyboxuila:CaoNgodbs2025@aws-0-us-west-2.pooler.supabase.com:6543/postgres"
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Fix Railway's postgres:// URL
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine and session
engine = get_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize Gemini
vision_model = None
if GEMINI_AVAILABLE:
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            vision_model = genai.GenerativeModel('gemini-2.0-flash')
            print(f"Gemini initialized successfully")
        else:
            print("Warning: GEMINI_API_KEY not found in environment")
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        vision_model = None

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Jewelry Recommendation API",
    description="K-Modes ML + SQLAlchemy ORM powered jewelry recommendations",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DEPENDENCY
# ============================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class FrontendRecommendationRequest(BaseModel):
    image: Optional[str] = Field(default=None, description="Base64 encoded image")
    preferences: dict = Field(
        default={},
        example={
            "styles": ["romantic", "classic"],
            "budgetMin": 1000,
            "budgetMax": 10000,
            "material": "Gold"
        }
    )

class FrontendRecommendation(BaseModel):
    id: int
    ref: str
    name: str
    price: int
    image_url: Optional[str]
    match_score: float
    style_tags: List[str]

class FrontendResponse(BaseModel):
    recommendations: List[FrontendRecommendation]
    session_id: Optional[str] = None
    image_analysis: Optional[dict] = None

# ============================================================
# ML CONSTANTS (from jewelry_ml_model.py)
# ============================================================

# Scoring weights - your teammate's version
WEIGHT_STYLE = 0.55      # 55% - Direct style match
WEIGHT_COLOR = 0.30      # 30% - Color coordination  
WEIGHT_CLUSTER = 0.15    # 15% - Cluster-style alignment

STYLE_GEM_WEIGHT = 0.70
STYLE_BAND_WEIGHT = 0.30

# Style profiles from your teammate's ML model
STYLE_PROFILES = {
    'classic': {
        'gems': ['diamond', 'diamonds', 'sapphire', 'sapphires', 'pearl'],
        'bands': ['platinum', 'white gold'],
        'description': 'Essential Lines and Solitaire styles'
    },
    'modern': {
        'gems': ['onyx', 'obsidians', 'black lacquer'],
        'bands': ['white gold', 'ceramic', 'black ceramic'],
        'description': 'Sleek, high-contrast monochrome'
    },
    'romantic': {
        'gems': ['pink sapphire', 'pink sapphires', 'rubies', 'spinels'],
        'bands': ['pink gold'],
        'description': 'Love and Trinity in rose hues'
    },
    'bold': {
        'gems': ['emeralds', 'tsavorite garnet', 'tsavorite garnets', 'rubies'],
        'bands': ['yellow gold', 'black lacquer'],
        'description': 'High-contrast vibrant statement pieces'
    },
    'bohemian': {
        'gems': ['malachite', 'lapis lazuli', 'chrysoprase', 'carnelian', 'amazonite'],
        'bands': ['yellow gold'],
        'description': 'Earthy, colorful semi-precious stones'
    },
    'minimalist': {
        'gems': [],
        'bands': ['white gold', 'platinum', 'pink gold'],
        'description': 'Clean metal-only bands'
    },
    'vintage': {
        'gems': ['amethyst', 'amethysts', 'citrine', 'peridots', 'mother_of_pearl'],
        'bands': ['yellow gold'],
        'description': 'Yellow gold with warm, multi-colored gems'
    },
    'luxurious': {
        'gems': ['diamonds'],
        'bands': ['platinum'],
        'description': 'High Jewelry / Haute Joaillerie'
    }
}

# Color mappings
GEM_COLORS = {
    'amazonite': 'green', 'amethyst': 'purple', 'amethysts': 'purple',
    'aquamarines': 'blue', 'aventurine': 'green',
    'carnelian': 'orange', 'carnelians': 'orange',
    'chrysoprase': 'green', 'chrysoprases': 'green',
    'citrine': 'yellow', 'coral': 'red',
    'diamond': 'white', 'diamonds': 'white', 'brown diamonds': 'brown',
    'emeralds': 'green', 'garnets': 'red', 'lapis lazuli': 'blue',
    'malachite': 'green', 'obsidians': 'black', 'onyx': 'black',
    'pearl': 'white', 'mother_of_pearl': 'gray', 'gray mother_of_pearl': 'gray',
    'white mother_of_pearl': 'iridescent white', 'peridots': 'green',
    'pink sapphire': 'pink', 'pink sapphires': 'pink',
    'rubies': 'red', 'sapphire': 'blue', 'sapphires': 'blue',
    'spessartite garnet': 'orange', 'spinels': 'red',
    'tsavorite garnet': 'green', 'tsavorite garnets': 'green'
}

BAND_COLORS = {
    'pink gold': 'pink', 'white gold': 'silver', 'yellow gold': 'gold',
    'platinum': 'silver', 'non_rhodiumized white gold': 'silver',
    'ceramic': 'white', 'black ceramic': 'black',
    'lacquer': 'red', 'black lacquer': 'black'
}

MATERIAL_TO_COLORS = {
    'Gold': ['gold', 'yellow'],
    'Silver': ['silver', 'white'],
    'Platinum': ['silver', 'white'],
    'Rose Gold': ['pink', 'gold']
}

# ============================================================
# IMAGE ANALYSIS using Gemini Vision
# ============================================================

def analyze_outfit_image(image_data: str) -> Dict:
    """Analyze outfit image using Gemini Vision API"""
    if not vision_model:
        return {'colors': [], 'styles': []}
    
    try:
        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        prompt = """Analyze this outfit image and extract:
1. Dominant colors (choose from: black, white, gray, silver, gold, pink, red, orange, yellow, green, blue, purple, brown, beige, neutral)
2. Style characteristics (choose from: classic, modern, romantic, bold, bohemian, minimalist, vintage, luxurious)

Return ONLY a JSON object with this exact format:
{
  "colors": ["color1", "color2", "color3"],
  "styles": ["style1", "style2"]
}

Be specific and choose 2-4 colors and 1-3 styles that best match the outfit."""
        
        response = vision_model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        
        response_text = response.text.strip()
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]
        
        result = json.loads(response_text)
        
        return {
            'colors': [c.lower() for c in result.get('colors', [])],
            'styles': [s.lower() for s in result.get('styles', [])]
        }
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return {'colors': [], 'styles': []}

# ============================================================
# RECOMMENDATION ENGINE (K-Modes Logic + ORM)
# ============================================================

class RecommendationEngine:
    """
    K-Modes based recommendation engine using ORM
    
    Scoring: Total (0-10) = Style (55%) + Color (30%) + Cluster (15%)
    """
    
    def __init__(self, db: Session):
        self.db = db
        self._load_cluster_styles()
    
    def _load_cluster_styles(self):
        """Load cluster style assignments from database"""
        self.cluster_styles = {}
        clusters = self.db.query(Cluster).all()
        
        for cluster in clusters:
            self.cluster_styles[cluster.id] = {
                'primary_style': cluster.primary_style,
                'style_confidence': float(cluster.style_confidence) if cluster.style_confidence else 0.5,
                'dominant_colors': cluster.dominant_colors or []
            }
    
    def get_item_materials(self, item: JewelryItem) -> Tuple[set, set]:
        """Get gems and bands for an item using ORM"""
        gems = set()
        bands = set()
        
        for im in item.materials:
            if im.material.type == 'gem':
                gems.add(im.material.name)
            else:
                bands.add(im.material.name)
        
        return gems, bands
    
    def get_item_colors(self, item: JewelryItem) -> List[str]:
        """Get colors for an item based on its materials"""
        colors = set()
        
        for im in item.materials:
            mat_name = im.material.name
            if mat_name in GEM_COLORS:
                colors.add(GEM_COLORS[mat_name])
            elif mat_name in BAND_COLORS:
                colors.add(BAND_COLORS[mat_name])
        
        return list(colors) if colors else ['neutral']
    
    def calculate_style_score(self, item: JewelryItem, preferred_styles: List[str]) -> float:
        """
        Calculate style match score using your teammate's logic
        
        Returns: 0 to 5.5 points (55% weight * 10 max)
        """
        if not preferred_styles:
            return 0.0
        
        item_gems, item_bands = self.get_item_materials(item)
        
        # Build target materials from preferred styles
        target_gems = set()
        target_bands = set()
        for style in preferred_styles:
            if style in STYLE_PROFILES:
                target_gems.update(STYLE_PROFILES[style]['gems'])
                target_bands.update(STYLE_PROFILES[style]['bands'])
        
        # Calculate matches
        gem_matches = len(item_gems & target_gems)
        band_matches = len(item_bands & target_bands)
        
        max_gems = len(target_gems) if target_gems else 1
        max_bands = len(target_bands) if target_bands else 1
        
        gem_prop = min(gem_matches / max_gems, 1.0)
        band_prop = min(band_matches / max_bands, 1.0)
        
        # Weighted combination
        raw = gem_prop * STYLE_GEM_WEIGHT + band_prop * STYLE_BAND_WEIGHT
        
        # Penalty for modern style with warm gold
        if 'modern' in preferred_styles:
            if 'yellow gold' in item_bands or 'pink gold' in item_bands:
                raw *= 0.6
        
        # Penalty for bohemian/vintage with diamonds
        item_has_diamonds = 'diamonds' in item_gems or 'diamond' in item_gems
        if any(s in ['bohemian', 'vintage', 'minimalist'] for s in preferred_styles) and item_has_diamonds:
            raw *= 0.8
        
        return raw * 10 * WEIGHT_STYLE
    
    def calculate_color_score(self, item: JewelryItem, preferred_colors: List[str]) -> float:
        """
        Calculate color match score
        
        Returns: 0 to 3.0 points (30% weight * 10 max)
        """
        if not preferred_colors:
            return 0.0
        
        item_colors = self.get_item_colors(item)
        matches = len(set(item_colors) & set(preferred_colors))
        raw = min(matches / len(preferred_colors), 1.0)
        
        return raw * 10 * WEIGHT_COLOR
    
    def calculate_cluster_bonus(self, item: JewelryItem, 
                                 preferred_styles: List[str],
                                 preferred_colors: List[str]) -> float:
        """
        Calculate cluster-style alignment bonus
        
        Returns: 0 to 1.5 points (15% weight * 10 max)
        """
        if not item.cluster_id or item.cluster_id not in self.cluster_styles:
            return 0.0
        
        cluster_info = self.cluster_styles[item.cluster_id]
        
        # Style alignment (50% of cluster bonus)
        style_bonus = 0.0
        if preferred_styles and cluster_info['primary_style'] in preferred_styles:
            style_bonus = cluster_info['style_confidence']
        
        # Color alignment (50% of cluster bonus)
        color_bonus = 0.0
        if preferred_colors and cluster_info['dominant_colors']:
            cluster_colors = cluster_info['dominant_colors'][:3]
            overlap = len(set(cluster_colors) & set(preferred_colors))
            max_overlap = min(len(cluster_colors), len(preferred_colors))
            color_bonus = overlap / max_overlap if max_overlap > 0 else 0
        
        raw_bonus = (style_bonus * 0.5 + color_bonus * 0.5)
        return raw_bonus * 10 * WEIGHT_CLUSTER
    
    def get_recommendations(
        self,
        preferred_styles: List[str],
        preferred_colors: List[str],
        min_price: int,
        max_price: int,
        limit: int = 6
    ) -> List[dict]:
        """Generate recommendations using ORM with K-Modes scoring"""
        
        # Query with ORM - filter by price and stock
        query = self.db.query(JewelryItem).join(Inventory).filter(
            Inventory.status == 'active',
            Inventory.stock_qty > 0,
            JewelryItem.price >= min_price,
            JewelryItem.price <= max_price
        )
        
        items = query.all()
        
        if not items:
            return []
        
        # Score each item
        scored_items = []
        for item in items:
            style_score = self.calculate_style_score(item, preferred_styles)
            color_score = self.calculate_color_score(item, preferred_colors)
            cluster_bonus = self.calculate_cluster_bonus(item, preferred_styles, preferred_colors)
            total_score = style_score + color_score + cluster_bonus
            
            scored_items.append({
                'item': item,
                'style_score': round(style_score, 2),
                'color_score': round(color_score, 2),
                'cluster_bonus': round(cluster_bonus, 2),
                'total_score': round(total_score, 2)
            })
        
        # Sort by score descending
        scored_items.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Build results
        results = []
        for rank, scored in enumerate(scored_items[:limit], 1):
            item = scored['item']
            results.append({
                'rank': rank,
                'item_id': item.id,
                'ref': item.ref,
                'title': item.title,
                'price': item.price,
                'image_url': item.image_url,
                'cluster_style': item.cluster.primary_style if item.cluster else None,
                'total_score': scored['total_score'],
                'style_score': scored['style_score'],
                'color_score': scored['color_score'],
                'cluster_bonus': scored['cluster_bonus']
            })
        
        return results

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "healthy",
        "service": "Jewelry Recommendation API",
        "version": "2.0.0 (K-Modes + ORM)",
        "image_analysis": "enabled" if vision_model else "disabled",
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
def health_check(db: Session = Depends(get_db)):
    try:
        item_count = db.query(func.count(JewelryItem.id)).scalar()
        cluster_count = db.query(func.count(Cluster.id)).scalar()

        key = os.getenv("GEMINI_API_KEY") or ""
        key_stripped = key.strip()

        return {
            "status": "healthy",
            "database": "connected",
            "items": item_count,
            "clusters": cluster_count,
            "gemini_available": GEMINI_AVAILABLE,
            "gemini_key_exists": bool(key_stripped),
            "gemini_key_len": len(key_stripped),
            "vision_model_initialized": vision_model is not None,
            "image_analysis": "enabled" if vision_model else "disabled",
            "railway_service": os.getenv("RAILWAY_SERVICE_NAME"),
            "railway_env": os.getenv("RAILWAY_ENVIRONMENT"),
            "has_env_key": "GEMINI_API_KEY" in os.environ,
            "key_len": len((os.getenv("GEMINI_API_KEY") or "").strip()),
            "has_test_var": "TEST_VAR" in os.environ,
            "test_var": os.getenv("TEST_VAR"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# ------------------------------------------------------------
# MAIN RECOMMENDATION ENDPOINT
# ------------------------------------------------------------

@app.post("/get-recommendations", response_model=FrontendResponse, tags=["Recommendations"])
def get_recommendations_frontend(
    request: FrontendRecommendationRequest,
    db: Session = Depends(get_db)
):
    """
    Get jewelry recommendations with K-Modes ML scoring + ORM
    
    Scoring: Total (0-10) = Style (55%) + Color (30%) + Cluster (15%)
    """
    prefs = request.preferences
    
    # Step 1: Analyze image if provided
    image_analysis = {'colors': [], 'styles': []}
    if request.image and vision_model:
        image_analysis = analyze_outfit_image(request.image)
    
    # Step 2: Combine preferences
    user_styles = [s.lower() for s in prefs.get('styles', [])]
    all_styles = list(set(user_styles + image_analysis['styles']))
    
    # Use image colors if available, otherwise use material preference
    preferred_colors = image_analysis['colors']
    if not preferred_colors:
        material = prefs.get('material', '')
        if material and material in MATERIAL_TO_COLORS:
            preferred_colors = MATERIAL_TO_COLORS[material]
    
    budget_min = prefs.get('budgetMin', 0)
    budget_max = prefs.get('budgetMax', 500000)
    
    # Step 3: Get recommendations using K-Modes scoring + ORM
    rec_engine = RecommendationEngine(db)
    results = rec_engine.get_recommendations(
        preferred_styles=all_styles,
        preferred_colors=preferred_colors,
        min_price=budget_min,
        max_price=budget_max,
        limit=6
    )
    
    # Step 4: Log session and recommendations (with error handling)
    session_id = uuid.uuid4()
    
    try:
        outfit_session = OutfitSession(
            id=session_id,
            preferred_styles=all_styles,
            preferred_colors=preferred_colors,
            min_price=budget_min,
            max_price=budget_max,
            clothing_features=image_analysis if image_analysis['colors'] else {}
        )
        db.add(outfit_session)
        db.flush()
        
        for r in results:
            log = RecommendationLog(
                session_id=session_id,
                item_id=r['item_id'],
                rank=r['rank'],
                total_score=r['total_score'],
                style_score=r['style_score'],
                color_score=r['color_score'],
                cluster_bonus=r['cluster_bonus'],
                cluster_style=r['cluster_style']
            )
            db.add(log)
        
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Warning: Failed to log recommendations: {e}")
    
    # Step 5: Format response for frontend
    recommendations = []
    for r in results:
        match_score = min(0.50 + (r['total_score'] / 10.0), 1.0)
        
        style_tags = []
        if r['cluster_style']:
            style_tags.append(r['cluster_style'])
        
        recommendations.append(FrontendRecommendation(
            id=r['item_id'],
            ref=r['ref'],
            name=r['title'],
            price=r['price'],
            image_url=r['image_url'],
            match_score=round(match_score, 2),
            style_tags=style_tags
        ))
    
    return FrontendResponse(
        recommendations=recommendations,
        session_id=str(session_id),
        image_analysis=image_analysis if image_analysis['colors'] else None
    )

# ------------------------------------------------------------
# ITEMS ENDPOINT
# ------------------------------------------------------------

@app.get("/items", tags=["Items"])
def list_items(
    skip: int = 0,
    limit: int = 20,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    db: Session = Depends(get_db)
):
    query = self.db.query(JewelryItem).join(Inventory).filter(
        Inventory.status == 'active',
        Inventory.stock_qty > 0
    )
    
    if min_price:
        query = query.filter(JewelryItem.price >= min_price)
    if max_price:
        query = query.filter(JewelryItem.price <= max_price)
    
    items = query.offset(skip).limit(limit).all()
    
    return {
        "items": [item.to_dict() for item in items]
    }

# ------------------------------------------------------------
# CLUSTERS ENDPOINT
# ------------------------------------------------------------

@app.get("/clusters", tags=["Clusters"])
def list_clusters(db: Session = Depends(get_db)):
    clusters = db.query(
        Cluster,
        func.count(JewelryItem.id).label('item_count'),
        func.avg(JewelryItem.price).label('avg_price')
    ).outerjoin(JewelryItem).group_by(Cluster.id).all()
    
    return [
        {
            "id": c.id,
            "primary_style": c.primary_style,
            "style_confidence": float(c.style_confidence) if c.style_confidence else None,
            "item_count": count,
            "avg_price": float(avg) if avg else None,
            "dominant_colors": c.dominant_colors
        }
        for c, count, avg in clusters
    ]

# ------------------------------------------------------------
# ANALYTICS ENDPOINT
# ------------------------------------------------------------

@app.get("/analytics/overview", tags=["Analytics"])
def get_analytics(db: Session = Depends(get_db)):
    total_items = db.query(func.count(JewelryItem.id)).scalar()
    total_sessions = db.query(func.count(OutfitSession.id)).scalar()
    total_recs = db.query(func.count(RecommendationLog.id)).scalar()
    total_clicks = db.query(func.count(RecommendationLog.id)).filter(
        RecommendationLog.clicked == True
    ).scalar()
    
    return {
        "total_items": total_items,
        "total_sessions": total_sessions,
        "total_recommendations": total_recs,
        "total_clicks": total_clicks,
        "ctr": round(total_clicks / total_recs * 100, 2) if total_recs > 0 else 0
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
