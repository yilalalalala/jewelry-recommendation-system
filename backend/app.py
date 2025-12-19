"""
Jewelry Recommendation API - FastAPI Application

"""

import os
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid

from sqlalchemy import create_engine, func, and_, or_, Integer
from sqlalchemy.orm import sessionmaker, Session

from models import (
    Base, Category, Material, Cluster, JewelryItem, 
    ItemMaterial, Inventory, UserProfile, OutfitSession, 
    RecommendationLog, get_engine
)

# CONFIGURATION - Uses Environment Variables

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres.onvusagrrakeyboxuila:CaoNgodbs2025@aws-0-us-west-2.pooler.supabase.com:6543/postgres"
)

# Railway uses DATABASE_URL with postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine and session factory
engine = get_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FASTAPI APP SETUP

app = FastAPI(
    title="Jewelry Recommendation API",
    description="AI-powered jewelry recommendations using K-Modes clustering and SQLAlchemy ORM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - Allow frontend access
# Update with your actual Vercel domain
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://*.vercel.app",
    "*"  # For development - restrict in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DEPENDENCY: Database Session

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# PYDANTIC SCHEMAS

class RecommendationRequest(BaseModel):
    """Request schema for getting recommendations"""
    preferred_styles: List[str] = Field(default=[], example=["romantic", "classic"])
    preferred_colors: List[str] = Field(default=[], example=["pink", "white"])
    min_price: int = Field(default=0, ge=0, example=1000)
    max_price: int = Field(default=500000, ge=0, example=10000)
    category: Optional[str] = Field(default=None, example="rings")
    limit: int = Field(default=10, ge=1, le=50, example=10)
    user_id: Optional[str] = Field(default=None)


class FrontendRecommendationRequest(BaseModel):
    """Request schema matching React frontend format"""
    image: Optional[str] = Field(default=None, description="Base64 encoded image (optional)")
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
    """Response format matching React frontend expectations"""
    id: int
    ref: str
    name: str
    price: int
    image_url: Optional[str]
    match_score: float  # 0-1 scale for frontend display
    style_tags: List[str]


class FrontendResponse(BaseModel):
    """Response wrapper for frontend"""
    recommendations: List[FrontendRecommendation]
    session_id: Optional[str] = None


# STYLE PROFILES (from ML model)

STYLE_PROFILES = {
    'classic': {
        'gems': ['diamond', 'pearl', 'sapphire', 'sapphires'],
        'bands': ['white gold', 'platinum']
    },
    'modern': {
        'gems': ['brown diamonds', 'obsidians', 'aquamarines', 'tsavorite garnet', 'onyx'],
        'bands': ['white gold', 'platinum', 'ceramic', 'black ceramic']
    },
    'romantic': {
        'gems': ['pink sapphire', 'pink sapphires', 'rubies', 'coral', 'pearl', 'spinels'],
        'bands': ['pink gold']
    },
    'bold': {
        'gems': ['emeralds', 'rubies', 'sapphire', 'sapphires', 'garnets', 'peridots'],
        'bands': ['yellow gold', 'black lacquer']
    },
    'bohemian': {
        'gems': ['amazonite', 'aventurine', 'carnelian', 'malachite', 'chrysoprase', 'lapis lazuli'],
        'bands': ['yellow gold', 'pink gold']
    },
    'minimalist': {
        'gems': ['aquamarines', 'carnelian', 'amethyst'],
        'bands': ['white gold', 'platinum']
    },
    'vintage': {
        'gems': ['pearl', 'amethyst', 'amethysts', 'citrine', 'peridots', 'mother_of_pearl'],
        'bands': ['yellow gold', 'pink gold', 'black lacquer']
    },
    'luxurious': {
        'gems': ['diamond', 'diamonds'],
        'bands': ['platinum', 'white gold']
    }
}

COLOR_MAPPING = {
    'pink': ['pink', 'red', 'coral'],
    'red': ['red', 'pink', 'coral'],
    'white': ['white', 'silver', 'iridescent white'],
    'silver': ['silver', 'white', 'gray'],
    'gold': ['gold', 'yellow'],
    'black': ['black', 'brown'],
    'blue': ['blue'],
    'green': ['green'],
    'purple': ['purple'],
    'orange': ['orange', 'yellow'],
    'brown': ['brown', 'black']
}

# Map frontend materials to colors
MATERIAL_TO_COLOR = {
    'Gold': ['gold', 'yellow'],
    'Silver': ['silver', 'white'],
    'Platinum': ['silver', 'white'],
    'Rose Gold': ['pink', 'gold']
}


# RECOMMENDATION ENGINE (Using ORM)

class RecommendationEngine:
    """ORM-based recommendation engine"""
    
    WEIGHT_STYLE = 0.40
    WEIGHT_COLOR = 0.30
    WEIGHT_CLUSTER = 0.30
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_style_score(self, item: JewelryItem, preferred_styles: List[str]) -> float:
        """Calculate style match score (0-4 points)"""
        if not preferred_styles:
            return 0.0
        
        item_gems = set()
        item_bands = set()
        for im in item.materials:
            if im.material.type == 'gem':
                item_gems.add(im.material.name)
            else:
                item_bands.add(im.material.name)
        
        target_gems = set()
        target_bands = set()
        for style in preferred_styles:
            if style in STYLE_PROFILES:
                target_gems.update(STYLE_PROFILES[style]['gems'])
                target_bands.update(STYLE_PROFILES[style]['bands'])
        
        gem_matches = len(item_gems & target_gems)
        band_matches = len(item_bands & target_bands)
        
        max_gems = len(target_gems) if target_gems else 1
        max_bands = len(target_bands) if target_bands else 1
        
        gem_score = min(gem_matches / max_gems, 1.0)
        band_score = min(band_matches / max_bands, 1.0)
        
        return (gem_score * 0.6 + band_score * 0.4) * 10 * self.WEIGHT_STYLE
    
    def calculate_color_score(self, item: JewelryItem, preferred_colors: List[str]) -> float:
        """Calculate color match score (0-3 points)"""
        if not preferred_colors:
            return 0.0
        
        item_colors = set()
        for im in item.materials:
            if im.material.color:
                item_colors.add(im.material.color)
        
        target_colors = set()
        for color in preferred_colors:
            target_colors.add(color)
            if color in COLOR_MAPPING:
                target_colors.update(COLOR_MAPPING[color])
        
        matches = len(item_colors & target_colors)
        return min(matches / len(preferred_colors), 1.0) * 10 * self.WEIGHT_COLOR
    
    def calculate_cluster_bonus(self, item: JewelryItem, preferred_styles: List[str]) -> float:
        """Calculate cluster-style alignment bonus (0-3 points)"""
        if not item.cluster or not preferred_styles:
            return 0.0
        
        cluster_style = item.cluster.primary_style
        if cluster_style in preferred_styles:
            confidence = float(item.cluster.style_confidence) if item.cluster.style_confidence else 0.5
            return confidence * 10 * self.WEIGHT_CLUSTER
        
        return 0.0
    
    def get_recommendations(
        self,
        preferred_styles: List[str],
        preferred_colors: List[str],
        min_price: int,
        max_price: int,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[dict]:
        """Generate recommendations using ORM"""
        
        query = self.db.query(JewelryItem).join(Inventory).filter(
            Inventory.status == 'active',
            Inventory.stock_qty > 0,
            JewelryItem.price >= min_price,
            JewelryItem.price <= max_price
        )
        
        if category:
            query = query.join(Category).filter(Category.name == category)
        
        items = query.all()
        
        if not items:
            return []
        
        scored_items = []
        for item in items:
            style_score = self.calculate_style_score(item, preferred_styles)
            color_score = self.calculate_color_score(item, preferred_colors)
            cluster_bonus = self.calculate_cluster_bonus(item, preferred_styles)
            total_score = style_score + color_score + cluster_bonus
            
            scored_items.append({
                'item': item,
                'style_score': round(style_score, 2),
                'color_score': round(color_score, 2),
                'cluster_bonus': round(cluster_bonus, 2),
                'total_score': round(total_score, 2)
            })
        
        scored_items.sort(key=lambda x: x['total_score'], reverse=True)
        
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
                'category': item.category.name if item.category else None,
                'collection': item.collection,
                'cluster_style': item.cluster.primary_style if item.cluster else None,
                'total_score': scored['total_score'],
                'style_score': scored['style_score'],
                'color_score': scored['color_score'],
                'cluster_bonus': scored['cluster_bonus'],
                'in_stock': True
            })
        
        return results


# API ENDPOINTS

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Jewelry Recommendation API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check(db: Session = Depends(get_db)):
    """Detailed health check with database connectivity"""
    try:
        item_count = db.query(func.count(JewelryItem.id)).scalar()
        cluster_count = db.query(func.count(Cluster.id)).scalar()
        
        return {
            "status": "healthy",
            "database": "connected",
            "items": item_count,
            "clusters": cluster_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# FRONTEND-COMPATIBLE ENDPOINT (Main integration point)

@app.post("/get-recommendations", response_model=FrontendResponse, tags=["Frontend"])
def get_recommendations_frontend(
    request: FrontendRecommendationRequest,
    db: Session = Depends(get_db)
):
    """
    Frontend-compatible recommendation endpoint.
    
    Accepts the format sent by the React app:
    {
        "image": "base64...",  // optional
        "preferences": {
            "styles": ["romantic", "classic"],
            "budgetMin": 1000,
            "budgetMax": 10000,
            "material": "Gold"
        }
    }
    
    Returns format expected by React app:
    {
        "recommendations": [
            {
                "id": 1,
                "ref": "B4087800",
                "name": "Love ring",
                "price": 2760,
                "image_url": "https://...",
                "match_score": 0.75,
                "style_tags": ["romantic"]
            }
        ]
    }
    """
    prefs = request.preferences
    
    # Extract preferences from frontend format
    styles = prefs.get('styles', [])
    budget_min = prefs.get('budgetMin', 0)
    budget_max = prefs.get('budgetMax', 500000)
    material = prefs.get('material', '')
    
    # Convert material preference to colors
    colors = []
    if material and material in MATERIAL_TO_COLOR:
        colors = MATERIAL_TO_COLOR[material]
    
    # Get recommendations using ORM engine
    engine = RecommendationEngine(db)
    results = engine.get_recommendations(
        preferred_styles=styles,
        preferred_colors=colors,
        min_price=budget_min,
        max_price=budget_max,
        limit=6
    )
    
    # Create outfit session (required for foreign key)
    session_id = uuid.uuid4()
    
    try:
        outfit_session = OutfitSession(
            id=session_id,
            preferred_styles=styles,
            preferred_colors=colors,
            min_price=budget_min,
            max_price=budget_max
        )
        db.add(outfit_session)
        db.flush()
        
        # Log recommendations
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
    
    # Convert to frontend format
    recommendations = []
    for r in results:
        match_score = min(r['total_score'] / 10.0, 1.0)
        
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
        session_id=str(session_id)
    )


# ORIGINAL API ENDPOINT (for direct API usage)

@app.post("/recommendations", tags=["Recommendations"])
def get_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """
    Get jewelry recommendations based on preferences.
    Uses SQLAlchemy ORM for all database operations.
    """
    engine = RecommendationEngine(db)
    results = engine.get_recommendations(
        preferred_styles=request.preferred_styles,
        preferred_colors=request.preferred_colors,
        min_price=request.min_price,
        max_price=request.max_price,
        category=request.category,
        limit=request.limit
    )
    
    session_id = str(uuid.uuid4())
    
    for r in results:
        log = RecommendationLog(
            session_id=uuid.UUID(session_id),
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
    
    return {
        "session_id": session_id,
        "count": len(results),
        "recommendations": results
    }


# ITEMS ENDPOINTS

@app.get("/items", tags=["Items"])
def list_items(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    in_stock: bool = True,
    db: Session = Depends(get_db)
):
    """List jewelry items with filtering"""
    query = db.query(JewelryItem)
    
    if in_stock:
        query = query.join(Inventory).filter(
            Inventory.status == 'active',
            Inventory.stock_qty > 0
        )
    
    if category:
        query = query.join(Category).filter(Category.name == category)
    
    if min_price is not None:
        query = query.filter(JewelryItem.price >= min_price)
    
    if max_price is not None:
        query = query.filter(JewelryItem.price <= max_price)
    
    total = query.count()
    items = query.offset(skip).limit(limit).all()
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "items": [item.to_dict() for item in items]
    }


@app.get("/items/{item_id}", tags=["Items"])
def get_item(item_id: int, db: Session = Depends(get_db)):
    """Get item details by ID"""
    item = db.query(JewelryItem).filter(JewelryItem.id == item_id).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    inventory = db.query(Inventory).filter(Inventory.item_id == item_id).first()
    
    return {
        **item.to_dict(),
        "inventory": inventory.to_dict() if inventory else None,
        "materials": [
            {
                "name": im.material.name,
                "type": im.material.type,
                "color": im.material.color,
                "quantity": im.quantity
            }
            for im in item.materials
        ]
    }


# CLUSTERS ENDPOINTS

@app.get("/clusters", tags=["Clusters"])
def list_clusters(db: Session = Depends(get_db)):
    """List all style clusters"""
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


# ANALYTICS ENDPOINTS

@app.get("/analytics/overview", tags=["Analytics"])
def get_analytics_overview(db: Session = Depends(get_db)):
    """Get overall analytics summary"""
    total_items = db.query(func.count(JewelryItem.id)).scalar()
    total_users = db.query(func.count(UserProfile.id)).scalar()
    total_sessions = db.query(func.count(OutfitSession.id)).scalar()
    total_recs = db.query(func.count(RecommendationLog.id)).scalar()
    total_clicks = db.query(func.count(RecommendationLog.id)).filter(
        RecommendationLog.clicked == True
    ).scalar()
    total_purchases = db.query(func.count(RecommendationLog.id)).filter(
        RecommendationLog.purchased == True
    ).scalar()
    
    ctr = (total_clicks / total_recs * 100) if total_recs > 0 else 0
    cvr = (total_purchases / total_clicks * 100) if total_clicks > 0 else 0
    
    return {
        "total_items": total_items,
        "total_users": total_users,
        "total_sessions": total_sessions,
        "total_recommendations": total_recs,
        "total_clicks": total_clicks,
        "total_purchases": total_purchases,
        "click_through_rate": round(ctr, 2),
        "conversion_rate": round(cvr, 2)
    }


# MAIN

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
