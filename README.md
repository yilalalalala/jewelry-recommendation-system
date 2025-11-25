# Jewelry Recommendation System

Agentic AI + E-Commerce Database for Jewelry Outfit Styling and Sales Uplift

## Overview

This project implements an intelligent jewelry recommendation system that combines agentic AI with a traditional e-commerce database. The system helps customers style jewelry with their outfits while improving jewelry sales conversion by recommending only in-stock items.

## Project Goals

1. **Improve Customer Experience**: Help customers make better styling decisions and reduce choice overload
2. **Increase Sales Conversion**: Recommend only available inventory, reducing frustration from out-of-stock items
3. **Gather Analytics**: Collect metrics on customer satisfaction and behavior for product improvement

## Features

- **Smart Filtering**: Multi-criteria filtering (budget, style, materials, category)
- **Intelligent Scoring**: Combines style overlap and price fit for optimal recommendations
- **Inventory-Aware**: Only recommends items that are actually in stock
- **Analytics Dashboard**: Track trends and popular styles for merchandising insights
- **ML-Based Recommendations**: Content-based filtering using TF-IDF and cosine similarity

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── example_usage.py
├── src/
│   ├── __init__.py
│   ├── recommendation_system.py    # Rule-based recommendation engine
│   └── ml_recommender.py           # ML-based recommendation engine
├── data/
│   └── ready_for_ml_adjusted.csv   # Jewelry inventory data (692 items)
├── notebooks/
│   └── Ruby_draft.ipynb            # Development notebooks and experiments
├── docs/
│   └── project_proposal_Cao_Ngo.pdf
└── tests/
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd project\ directory
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Rule-Based Recommendation System

```python
from src.recommendation_system import JewelryRecommendationSystem

# Initialize system
system = JewelryRecommendationSystem()
system.load_data('data/ready_for_ml_adjusted.csv')

# Get recommendations
recommendations = system.recommend_jewelry(
    budget_min=1000,
    budget_max=2000,
    style_preferences=['elegant', 'modern'],
    category='earrings',
    material_preferences=['diamond', 'white gold'],
    top_k=6
)

# Display results
for rec in recommendations:
    print(f"{rec['title']} - ${rec['price']}")
```

### ML-Based Recommendation System

```python
from src.ml_recommender import MLBasedRecommender

# Initialize ML recommender
recommender = MLBasedRecommender('data/ready_for_ml_adjusted.csv')

# Get recommendations
recommendations = recommender.recommend(
    theme="Romantic Date Night",
    preferences_text="I am wearing a green dress and like yellow metal",
    item_type_focus="rings",
    price_min=1000,
    price_max=2000,
    top_k=5
)

print(recommendations)
```

### Run Example Script

```bash
python example_usage.py
```

## Data Schema

### Input CSV Format

The system expects jewelry data with the following columns:
- `ref`: Product reference number
- `title`: Product name/title
- `price`: Price in dollars
- `image`: URL to product image
- `quantity`: Available stock quantity
- Category columns: `rings`, `earrings`, `necklaces`, `bracelets` (binary)
- Material columns: `diamond`, `pearl`, `white gold`, `yellow gold`, etc. (binary)

### Database Schema Framework (Proposed)

**Relational (MySQL/RDS)**
- `JewelryItem(item_id PK, name, category, material, color, style_tags, price, image_url)`
- `Inventory(item_id PK/FK, stock_qty, status)`
- `RecommendationLog(rec_id PK, session_id, item_id FK, rank, created_at)`

**Document (MongoDB Atlas)**
- `UserProfile`: Store user preferences and history
- `OutfitSession`: Store user sessions with clothing features and inputs

## Recommendation Logic

### Hard Filters
Items must satisfy ALL of the following:
- ✓ In stock (quantity > 0)
- ✓ Within budget range
- ✓ Match category (if specified)
- ✓ Match material preferences (if specified)

### Scoring Algorithm

**Rule-Based System:**
```
final_score = style_overlap + (0.5 × price_fit)
```

Where:
- `style_overlap`: Number of matching style tags between user preferences and item
- `price_fit`: 1 - (|item_price - budget_midpoint| / budget_range)

**ML-Based System:**
```
final_score = similarity_score × stock_adjustment
```

Where:
- `similarity_score`: Cosine similarity between user vector and item features
- `stock_adjustment`: 0.9 for low stock items, 1.0 otherwise

## Use Cases

### For Customers
- Match jewelry to a specific outfit for an event
- Plan multiple outfits with coordinating jewelry
- Find jewelry for specific occasions (wedding, job interview, etc.)
- Discover new styles within budget constraints

### For Merchandising Teams
- Identify unmet demand patterns
- Track emerging fashion trends
- Implement dynamic pricing strategies
- Optimize inventory based on customer preferences

## Analytics

The system logs all recommendations and provides analytics:

```python
trends = system.analyze_trends()
```

Returns:
- Total sessions
- Popular style preferences
- Popular categories
- Average budget ranges

## Implementation Roadmap

- [x] Step 1: Create database schema and load data
- [x] Step 2: Implement recommendation algorithms (rule-based and ML)
- [x] Step 3: Add logging and analytics
- [ ] Step 4: Build web UI (Flask/React)
- [ ] Step 5: Add image analysis for outfit matching
- [ ] Step 6: Deploy demo with real-time recommendations

## Technologies Used

- **Python 3.8+**
- **pandas & numpy**: Data manipulation
- **scikit-learn**: ML algorithms (TF-IDF, cosine similarity)
- **Flask**: Web framework (future)
- **MySQL/MongoDB**: Database layer (future)

## Team

- Cao Yila
- Ngo [Last Name]

## License

This project is for educational purposes as part of a Database Systems course.

## References

- Project Proposal: See `docs/project_proposal_Cao_Ngo.pdf`
- Development Notebooks: See `notebooks/Ruby_draft.ipynb`

## Future Enhancements

- [ ] Image analysis for clothing color extraction
- [ ] Real-time inventory sync with database
- [ ] User authentication and profile management
- [ ] A/B testing framework for recommendation algorithms
- [ ] Mobile app integration
- [ ] Social sharing features
