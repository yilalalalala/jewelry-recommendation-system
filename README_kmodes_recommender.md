# Jewelry Recommendation System Using K-Modes Clustering

A machine learning-powered recommendation engine for jewelry e-commerce that uses K-Modes clustering to match customers with products based on style preferences, color choices, and budget constraints.

## Overview

This system helps customers find jewelry that matches their personal style while only recommending items that are in stock and within budget. It uses unsupervised learning (K-Modes clustering) to group jewelry items by material patterns, then applies a transparent 0-10 scoring system to rank recommendations.

### Key Features

- **K-Modes Clustering**: Groups 509 jewelry items into 8 style-labeled clusters based on 48 material features
- **Transparent Scoring**: 0-10 normalized scale combining Style (40%), Color (30%), and Cluster Bonus (30%)
- **Automatic Style Assignment**: Clusters are automatically labeled (Romantic, Bold, Luxurious, Modern, etc.)
- **Inventory-Aware**: Only recommends items that are in stock and within budget
- **Rarity-Weighted Matching**: Rare materials receive higher weight in style matching

## How It Works

### System Architecture

```
User Preferences (style, colors, budget)
    ↓
Hard Filters (price range, stock quantity)
    ↓
K-Modes Cluster Matching
    ↓
Scoring Engine (Style + Color + Cluster Bonus)
    ↓
Top K Recommendations
```

### Scoring System

| Component | Weight | Description |
|-----------|--------|-------------|
| **Style Score** | 40% | How well item materials match selected style profiles |
| **Color Score** | 30% | Direct color match between item and preferences |
| **Cluster Bonus** | 30% | Bonus if item's cluster aligns with preferred style |

### Style Profiles

The system supports 8 predefined style profiles:

| Style | Key Materials | Description |
|-------|---------------|-------------|
| Classic | Diamond, pearl, sapphire, white gold, platinum | Timeless elegance |
| Modern | Brown diamonds, obsidians, ceramic | Contemporary designs |
| Romantic | Pink sapphire, rubies, coral, pink gold | Soft, feminine touches |
| Bold | Emeralds, rubies, garnets, yellow gold | Statement pieces |
| Bohemian | Amazonite, malachite, lapis lazuli | Natural, earthy aesthetics |
| Minimalist | Aquamarines, white gold, platinum | Simple elegance |
| Vintage | Pearl, amethyst, citrine, yellow gold | Classic retro charm |
| Luxurious | Diamond, platinum, white gold | Premium diamond-focused |

## Quick Start

### Prerequisites

```bash
pip install pandas numpy kmodes
```

### Basic Usage

```python
from jewelry_recommender import JewelryDataLoader, JewelryRecommender

# Load data
loader = JewelryDataLoader('ready_for_ml_adjusted.csv')
loader.load()

# Train model
recommender = JewelryRecommender(loader, n_clusters=8)
recommender.fit()

# Get recommendations
results = recommender.recommend(
    preferred_colors=['pink', 'red'],
    preferred_styles=['romantic'],
    min_price=2000,
    max_price=4000,
    top_k=5
)

print(results[['ref', 'title', 'price', 'cluster_style', 'total_score']])
```

### Example Output

```
      ref                        title  price cluster_style  total_score
 B4087800                    Love ring   2760      romantic         5.95
 B8041700        C de Cartier earrings   2080      romantic         4.75
 B7215600 Diamants Légers necklace, LM   2390      romantic         4.47
 B7058700             Trinity necklace   2910      romantic         4.47
 B3153116   Hearts and Symbols pendant   2490      romantic         4.47
```

## Project Structure

```
jewelry-recommendation-system/
├── jewelry_recommender.py    # Main recommendation engine
├── ready_for_ml_adjusted.csv # Jewelry dataset (509 items)
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Data Schema

The dataset contains 509 jewelry items with 56 columns:

### Core Columns (7)
- `ref`: Product ID
- `title`: Product name
- `price`: Price in USD
- `image`: Product image URL
- `quantity`: Stock quantity
- `size`, `chain_size`: Sizing information

### Feature Columns (48 binary)
- **Categories (4)**: bracelets, earrings, necklaces, rings
- **Gemstones (34)**: diamond, pearl, ruby, emerald, sapphire, onyx, etc.
- **Bands/Metals (9)**: yellow gold, white gold, pink gold, platinum, etc.

## API Reference

### JewelryDataLoader

```python
loader = JewelryDataLoader(filepath='data.csv')
loader.load()                        # Load and preprocess data
loader.get_item_colors(row)          # Get colors from item materials
loader.get_clustering_features()     # Get feature matrix for clustering
```

### JewelryRecommender

```python
recommender = JewelryRecommender(data_loader, n_clusters=8)
recommender.fit()                    # Train K-Modes clustering

# Get recommendations
results = recommender.recommend(
    preferred_colors=['pink', 'white'],   # List of preferred colors
    preferred_styles=['romantic'],         # List of style profiles
    min_price=1000,                        # Minimum budget
    max_price=5000,                        # Maximum budget
    min_quantity=1,                        # Minimum stock
    top_k=10,                              # Number of results
    return_breakdown=False                 # Include score breakdown
)

# View cluster analysis
recommender.print_cluster_summary()
recommender.get_cluster_style_mapping()
```

## Model Performance

Based on 7 demo scenarios:

| Scenario | Style | Avg Score | Assessment |
|----------|-------|-----------|------------|
| Romantic + Pink | Romantic | 4.82 | Strong matches |
| Classic + White | Classic | 4.77 | Strong matches |
| Luxurious + White | Luxurious | 5.28 | Strong matches |
| Bohemian + Blue | Bohemian | 2.79 |  Limited options |
| Vintage + Black | Vintage | 2.94 |  Inventory gap |
| Bold + Red | Bold | 2.46 |  Inventory gap |
| Modern + Green | Modern | 1.25 | Significant gap |

## Color Mappings

### Gemstone → Color
| Color | Gemstones |
|-------|-----------|
| White | diamond, diamonds, pearl |
| Green | emeralds, malachite, amazonite, tsavorite garnet, peridots |
| Blue | sapphire, sapphires, lapis lazuli, aquamarines |
| Red | rubies, coral, garnets, spinels |
| Pink | pink sapphire, pink sapphires |
| Black | onyx, obsidians |
| Purple | amethyst, amethysts |
| Orange | carnelian, spessartite garnet |

### Band/Metal → Color
| Color | Materials |
|-------|-----------|
| Gold | yellow gold |
| Silver | white gold, platinum |
| Pink | pink gold |
| White | ceramic, mother of pearl |
| Black | black ceramic, black lacquer |

## Technical Details

### K-Modes Algorithm
- **Why K-Modes**: Data is categorical/binary (not suitable for K-Means)
- **Distance Metric**: Hamming distance (count of mismatches)
- **Centroid**: Mode (most frequent value) instead of mean
- **Initialization**: Huang method (frequency-based)
- **Parameters**: n_clusters=8, n_init=20

### Scoring Formula

```
Total Score = Style Score + Color Score + Cluster Bonus

Style Score = (gem_proportion × 0.60 + band_proportion × 0.40) × 10 × 0.40
Color Score = (color_matches / preferred_colors) × 10 × 0.30
Cluster Bonus = (style_alignment × 0.5 + color_alignment × 0.5) × 10 × 0.30
```

## License

MIT License

## Authors

- Yila Cao
- Ngo

## Acknowledgments

- Dataset: Cartier jewelry catalog
- Course: CSCI-GA.2433-001 Database Systems, NYU
