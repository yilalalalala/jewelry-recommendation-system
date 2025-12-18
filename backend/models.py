"""
SQLAlchemy ORM Models for Jewelry Recommendation System
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text,
    ForeignKey, ARRAY, create_engine, DECIMAL
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid

# Base class for all ORM models
Base = declarative_base()


# ============================================================
# CORE PRODUCT TABLES
# ============================================================

class Category(Base):
    """Product categories: rings, necklaces, bracelets, earrings"""
    __tablename__ = 'categories'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    items = relationship("JewelryItem", back_populates="category")
    
    def __repr__(self):
        return f"<Category(id={self.id}, name='{self.name}')>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description
        }


class Material(Base):
    """Materials: gems and bands (metals)"""
    __tablename__ = 'materials'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    type = Column(String(20), nullable=False)  # 'gem' or 'band'
    color = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    items = relationship("ItemMaterial", back_populates="material")
    
    def __repr__(self):
        return f"<Material(id={self.id}, name='{self.name}', type='{self.type}')>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'color': self.color
        }


class Cluster(Base):
    """K-Modes clusters with style assignments"""
    __tablename__ = 'clusters'
    
    id = Column(Integer, primary_key=True)
    primary_style = Column(String(50), nullable=False)
    style_confidence = Column(DECIMAL(5, 4))
    secondary_styles = Column(ARRAY(Text))
    avg_price = Column(DECIMAL(12, 2))
    item_count = Column(Integer)
    dominant_colors = Column(ARRAY(Text))
    top_gems = Column(ARRAY(Text))
    top_bands = Column(ARRAY(Text))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    items = relationship("JewelryItem", back_populates="cluster")
    
    def __repr__(self):
        return f"<Cluster(id={self.id}, style='{self.primary_style}')>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'primary_style': self.primary_style,
            'style_confidence': float(self.style_confidence) if self.style_confidence else None,
            'secondary_styles': self.secondary_styles,
            'avg_price': float(self.avg_price) if self.avg_price else None,
            'item_count': self.item_count,
            'dominant_colors': self.dominant_colors,
            'top_gems': self.top_gems,
            'top_bands': self.top_bands
        }


class JewelryItem(Base):
    """Main product table - 509 jewelry items"""
    __tablename__ = 'jewelry_items'
    
    id = Column(Integer, primary_key=True)
    ref = Column(String(20), unique=True, nullable=False)
    title = Column(String(255), nullable=False)
    price = Column(Integer, nullable=False)
    image_url = Column(Text)
    quantity = Column(Integer, default=0)
    size = Column(String(20))
    chain_size = Column(String(20))
    category_id = Column(Integer, ForeignKey('categories.id'))
    cluster_id = Column(Integer, ForeignKey('clusters.id'))
    collection = Column(String(100))
    price_tier = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    category = relationship("Category", back_populates="items")
    cluster = relationship("Cluster", back_populates="items")
    inventory = relationship("Inventory", back_populates="item", uselist=False)
    materials = relationship("ItemMaterial", back_populates="item")
    recommendations = relationship("RecommendationLog", back_populates="item")
    
    def __repr__(self):
        return f"<JewelryItem(ref='{self.ref}', title='{self.title}', price={self.price})>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'ref': self.ref,
            'title': self.title,
            'price': self.price,
            'image_url': self.image_url,
            'quantity': self.quantity,
            'size': self.size,
            'collection': self.collection,
            'price_tier': self.price_tier,
            'category': self.category.name if self.category else None,
            'cluster_style': self.cluster.primary_style if self.cluster else None,
            'in_stock': self.inventory.stock_qty > 0 if self.inventory else False
        }
    
    def to_dict_full(self):
        """Full details including materials"""
        data = self.to_dict()
        data['materials'] = {
            'gems': [im.material.name for im in self.materials if im.material.type == 'gem'],
            'bands': [im.material.name for im in self.materials if im.material.type == 'band']
        }
        data['colors'] = list(set([im.material.color for im in self.materials if im.material.color]))
        return data


class ItemMaterial(Base):
    """Junction table: items <-> materials"""
    __tablename__ = 'item_materials'
    
    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey('jewelry_items.id', ondelete='CASCADE'))
    material_id = Column(Integer, ForeignKey('materials.id', ondelete='CASCADE'))
    
    # Relationships
    item = relationship("JewelryItem", back_populates="materials")
    material = relationship("Material", back_populates="items")
    
    def __repr__(self):
        return f"<ItemMaterial(item_id={self.item_id}, material_id={self.material_id})>"


class Inventory(Base):
    """Stock tracking for items"""
    __tablename__ = 'inventory'
    
    item_id = Column(Integer, ForeignKey('jewelry_items.id', ondelete='CASCADE'), primary_key=True)
    stock_qty = Column(Integer, default=0)
    status = Column(String(20), default='active')
    reorder_level = Column(Integer, default=5)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    item = relationship("JewelryItem", back_populates="inventory")
    
    def __repr__(self):
        return f"<Inventory(item_id={self.item_id}, qty={self.stock_qty}, status='{self.status}')>"
    
    def to_dict(self):
        return {
            'item_id': self.item_id,
            'stock_qty': self.stock_qty,
            'status': self.status,
            'reorder_level': self.reorder_level
        }


# ============================================================
# USER & SESSION TABLES
# ============================================================

class UserProfile(Base):
    """User profiles with JSONB preferences"""
    __tablename__ = 'user_profiles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True)
    name = Column(String(100))
    preferences = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    sessions = relationship("OutfitSession", back_populates="user")
    
    def __repr__(self):
        return f"<UserProfile(id={self.id}, email='{self.email}')>"
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'email': self.email,
            'name': self.name,
            'preferences': self.preferences,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def get_favorite_styles(self):
        return self.preferences.get('favorite_styles', [])
    
    def get_preferred_colors(self):
        return self.preferences.get('preferred_colors', [])
    
    def get_budget_range(self):
        return (
            self.preferences.get('budget_min', 0),
            self.preferences.get('budget_max', 999999)
        )


class OutfitSession(Base):
    """Outfit/search sessions with user inputs"""
    __tablename__ = 'outfit_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('user_profiles.id'))
    preferred_colors = Column(ARRAY(Text))
    preferred_styles = Column(ARRAY(Text))
    min_price = Column(Integer)
    max_price = Column(Integer)
    min_quantity = Column(Integer, default=1)
    clothing_features = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("UserProfile", back_populates="sessions")
    recommendations = relationship("RecommendationLog", back_populates="session")
    
    def __repr__(self):
        return f"<OutfitSession(id={self.id}, styles={self.preferred_styles})>"
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'user_id': str(self.user_id) if self.user_id else None,
            'preferred_colors': self.preferred_colors,
            'preferred_styles': self.preferred_styles,
            'min_price': self.min_price,
            'max_price': self.max_price,
            'clothing_features': self.clothing_features,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class RecommendationLog(Base):
    """Track recommendations with scores and user interactions"""
    __tablename__ = 'recommendation_log'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('outfit_sessions.id'))
    item_id = Column(Integer, ForeignKey('jewelry_items.id'))
    rank = Column(Integer)
    total_score = Column(DECIMAL(5, 2))
    style_score = Column(DECIMAL(5, 2))
    color_score = Column(DECIMAL(5, 2))
    cluster_bonus = Column(DECIMAL(5, 2))
    cluster_style = Column(String(50))
    clicked = Column(Boolean, default=False)
    purchased = Column(Boolean, default=False)
    click_timestamp = Column(DateTime)
    purchase_timestamp = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("OutfitSession", back_populates="recommendations")
    item = relationship("JewelryItem", back_populates="recommendations")
    
    def __repr__(self):
        return f"<RecommendationLog(id={self.id}, item_id={self.item_id}, score={self.total_score})>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': str(self.session_id) if self.session_id else None,
            'item_id': self.item_id,
            'rank': self.rank,
            'total_score': float(self.total_score) if self.total_score else None,
            'style_score': float(self.style_score) if self.style_score else None,
            'color_score': float(self.color_score) if self.color_score else None,
            'cluster_bonus': float(self.cluster_bonus) if self.cluster_bonus else None,
            'cluster_style': self.cluster_style,
            'clicked': self.clicked,
            'purchased': self.purchased
        }


# ============================================================
# DATABASE CONNECTION HELPER
# ============================================================

def get_engine(database_url: str):
    """Create SQLAlchemy engine"""
    return create_engine(database_url, echo=False)


def get_session(engine):
    """Create a new database session"""
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(engine):
    """Create all tables (if not exist)"""
    Base.metadata.create_all(engine)


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Example connection
    DATABASE_URL = "postgresql://postgres.onvusagrrakeyboxuila:CaoNgodbs2025@aws-0-us-west-2.pooler.supabase.com:6543/postgres"
    
    engine = get_engine(DATABASE_URL)
    session = get_session(engine)
    
    # Query examples using ORM
    print("=== ORM Query Examples ===\n")
    
    # 1. Get all clusters
    clusters = session.query(Cluster).all()
    print("Clusters:")
    for c in clusters:
        print(f"  {c.id}: {c.primary_style} ({c.item_count} items)")
    
    # 2. Get items by style
    romantic_items = session.query(JewelryItem).join(Cluster).filter(
        Cluster.primary_style == 'romantic'
    ).limit(5).all()
    print("\nRomantic Items:")
    for item in romantic_items:
        print(f"  {item.ref}: {item.title} - ${item.price}")
    
    # 3. Get items within budget
    budget_items = session.query(JewelryItem).join(Inventory).filter(
        JewelryItem.price.between(2000, 5000),
        Inventory.status == 'active',
        Inventory.stock_qty > 0
    ).limit(5).all()
    print("\nItems $2000-$5000:")
    for item in budget_items:
        print(f"  {item.ref}: ${item.price}")
    
    session.close()
