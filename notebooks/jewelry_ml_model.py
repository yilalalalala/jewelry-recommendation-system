"""
Jewelry Recommendation System Using K-mode Clustering - With Cluster-to-Style Mapping

"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional

# Use the kmodes package
from kmodes.kmodes import KModes

class JewelryDataLoader:
    #Load and preprocess jewelry data

    CATEGORY_COLS = ['bracelets', 'earrings', 'necklaces', 'rings']

    GEM_COLS = [
        'amazonite', 'amethyst', 'amethysts', 'aquamarines', 'aventurine',
        'carnelian', 'carnelians', 'chrysoprase', 'chrysoprases', 'citrine',
        'coral', 'diamond', 'diamonds', 'emeralds', 'garnets', 'lapis lazuli',
        'malachite', 'obsidians', 'onyx', 'pearl', 'mother_of_pearl', 
        'gray mother_of_pearl', 'white mother_of_pearl', 'peridots', 'pink sapphire',
        'pink sapphires', 'rubies', 'sapphire', 'sapphires', 'spessartite garnet',
        'spinels', 'tsavorite garnet', 'tsavorite garnets', 'brown diamonds'
    ]

    BAND_COLS = [
        'pink gold', 'white gold', 'yellow gold', 'platinum',
        'non_rhodiumized white gold', 'ceramic', 'black ceramic',
        'lacquer', 'black lacquer'
    ]

    GEM_COLORS = {
        'amazonite': 'green', 'amethyst': 'purple', 'amethysts': 'purple',
        'aquamarines': 'blue', 'aventurine': 'green',
        'carnelian': 'orange', 'carnelians': 'orange',
        'chrysoprase': 'green', 'chrysoprases': 'green',
        'citrine': 'yellow', 'coral': 'red',
        'diamond': 'white', 'diamonds': 'white', 'brown diamonds': 'brown',
        'emeralds': 'green', 'garnets': 'red', 'lapis lazuli': 'blue',
        'malachite': 'green', 'obsidians': 'black', 'onyx': 'black',
        'pearl': 'white', 'mother_of_pearl' : 'gray', 'gray mother_of_pearl': 'gray', 
        'white mother_of_pearl' : "iridescent white", 'peridots': 'green',
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

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.clustering_features = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.filepath)
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df.loc[:, ~self.df.columns.str.match(r'^Unnamed')]
        return self.df

    def get_item_colors(self, row: pd.Series) -> List[str]:
        colors = set()
        for gem, color in self.GEM_COLORS.items():
            if gem in row.index and row.get(gem, 0) == 1:
                colors.add(color)
        for band, color in self.BAND_COLORS.items():
            if band in row.index and row.get(band, 0) == 1:
                colors.add(color)
        return list(colors) if colors else ['neutral']

    def get_clustering_features(self) -> np.ndarray:
        feature_cols = []
        for col in self.CATEGORY_COLS + self.GEM_COLS + self.BAND_COLS:
            if col in self.df.columns:
                feature_cols.append(col)
        self.clustering_features = feature_cols
        return self.df[feature_cols].values


class JewelryRecommender:
    """
    Recommendation engine with:
    - Interpretable 0-10 scoring
    - Cluster-to-Style mapping

    SCORING SYSTEM:
    Total = Style Score (55%) + Color Score (30%) + Cluster-Style Bonus (15%)
    """

    # Weights - cluster bonus now more important since it includes style
    WEIGHT_STYLE = 0.55       # 55% - Direct style match
    WEIGHT_COLOR = 0.30       # 30% - Color coordination
    WEIGHT_CLUSTER = 0.15     # 15% - Cluster-style alignment

    STYLE_GEM_WEIGHT = 0.70
    STYLE_BAND_WEIGHT = 0.30

    # Style profiles with gems and bands
    STYLE_PROFILES = {
        'classic': {
            'gems': ['diamond', 'diamonds', 'sapphire', 'sapphires', 'pearl'],
            'bands': ['platinum', 'white gold'],
            'description': 'Essential Lines and Solitaire styles'
        },
        'modern': {
            'gems': ['onyx', 'obsidians', 'black lacquer'], # Panthère signature
            'bands': ['white gold', 'ceramic', 'black ceramic'],
            'description': 'Sleek, high-contrast monochrome'
        },
        'romantic': {
            'gems': ['pink sapphire', 'pink sapphires', 'rubies', 'spinels'],
            'bands': ['pink gold'], # Pink gold is the defining feature here
            'description': 'Love and Trinity in rose hues'
        },
        'bold': {
            'gems': ['emeralds', 'tsavorite garnet', 'tsavorite garnets', 'rubies'],
            'bands': ['yellow gold', 'black lacquer'], 
            'description': 'High-contrast vibrant statement pieces'
        },
        'bohemian': {
            'gems': ['malachite', 'lapis lazuli', 'chrysoprase', 'carnelian', 'amazonite', 'turquoise'], 
            'bands': ['yellow gold'],
            'description': 'Earthy, colorful semi-precious stones'
        },
        'minimalist': {
            'gems': [], # Focused almost entirely on the metal work
            'bands': ['white gold', 'platinum', 'pink gold'],
            'description': 'Clean metal-only bands (e.g., standard Love/Juste un Clou)'
        },
        'vintage': {
            'gems': ['amethyst', 'amethysts', 'citrine', 'peridots', 'mother_of_pearl'],
            'bands': ['yellow gold'],
            'description': 'Yellow gold with warm, multi-colored gems'
        },
        'luxurious': {
            'gems': ['diamonds'], # Reserved for high-carat diamond density
            'bands': ['platinum'],
            'description': 'High Jewelry / Haute Joaillerie'
        }
    }

    def __init__(self, data_loader: JewelryDataLoader, n_clusters: int = 8):
        self.data_loader = data_loader
        self.n_clusters = n_clusters
        self.kmodes = None
        self.cluster_labels = None
        self.cluster_profiles = None
        self.cluster_styles = None  # Maps cluster_id -> style assignments

    def _calculate_material_weights(self):
      #Calculate inverse document frequency weights for materials
      total_items = len(self.data_loader.df)
      self.material_weights = {}
      
      # Calculate weights for gems
      for gem in self.data_loader.GEM_COLS:
          if gem in self.data_loader.df.columns:
              count = self.data_loader.df[gem].sum()
              if count > 0:
                  # Inverse frequency: rare materials get higher weight
                  self.material_weights[gem] = np.log(total_items / (count + 1))
      
      # Calculate weights for bands
      for band in self.data_loader.BAND_COLS:
          if band in self.data_loader.df.columns:
              count = self.data_loader.df[band].sum()
              if count > 0:
                  self.material_weights[band] = np.log(total_items / (count + 1))

    def fit(self) -> 'JewelryRecommender':
        #Train K-Modes clustering and assign styles to clusters
        features = self.data_loader.get_clustering_features()
        
        self.kmodes = KModes(
            n_clusters=self.n_clusters,
            init='Huang',
            n_init=20, # Number of different times algorithm will be run
            verbose=1,
            random_state=42
        )
        self.cluster_labels = self.kmodes.fit_predict(features)
        self.data_loader.df['cluster'] = self.cluster_labels
        
        self._calculate_material_weights()  # Add this line
        self._generate_cluster_profiles()
        self._assign_styles_to_clusters()
        
        return self

    def _assign_styles_to_clusters(self):
        """Assign styles using rarity-weighted scoring."""
        self.cluster_styles = {}

        for cluster_id, profile in self.cluster_profiles.items():
            style_scores = {}
            cluster_size = profile['size']

            for style_name, style_def in self.STYLE_PROFILES.items():
                # Weighted gem score
                gem_score = 0
                gem_weight_sum = 0
                for gem in style_def['gems']:
                    if gem in profile['gem_counts']:
                        frequency = profile['gem_counts'][gem] / cluster_size
                        rarity_weight = self.material_weights.get(gem, 1.0)
                        gem_score += frequency * rarity_weight
                        gem_weight_sum += rarity_weight
                
                gem_score = gem_score / gem_weight_sum if gem_weight_sum > 0 else 0

                # Weighted band score
                band_score = 0
                band_weight_sum = 0
                for band in style_def['bands']:
                    if band in profile['band_counts']:
                        frequency = profile['band_counts'][band] / cluster_size
                        rarity_weight = self.material_weights.get(band, 1.0)
                        band_score += frequency * rarity_weight
                        band_weight_sum += rarity_weight
                
                band_score = band_score / band_weight_sum if band_weight_sum > 0 else 0

                total_score = (gem_score * self.STYLE_GEM_WEIGHT +
                              band_score * self.STYLE_BAND_WEIGHT)

                style_scores[style_name] = round(total_score, 3)

            sorted_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)

            self.cluster_styles[cluster_id] = {
                'primary_style': sorted_styles[0][0] if sorted_styles[0][1] > 0 else 'neutral',
                'primary_score': sorted_styles[0][1],
                'all_styles': {s: score for s, score in sorted_styles if score > 0},
                'top_3_styles': sorted_styles[:3]
            }

    def _generate_cluster_profiles(self):
        #Generate descriptive profiles for each cluster
        self.cluster_profiles = {}

        for cluster_id in range(self.n_clusters):
            cluster_data = self.data_loader.df[self.data_loader.df['cluster'] == cluster_id]

            profile = {
                'size': len(cluster_data),
                'avg_price': cluster_data['price'].mean(),
                'categories': [],
                'top_gems': [],
                'top_bands': [],
                'dominant_colors': [],
                'gem_counts': {},   # For style matching
                'band_counts': {}   # For style matching
            }

            # Categories
            for cat in self.data_loader.CATEGORY_COLS:
                if cat in cluster_data.columns:
                    count = cluster_data[cat].sum()
                    if count > 0:
                        profile['categories'].append((cat, count))
            profile['categories'].sort(key=lambda x: x[1], reverse=True)

            # Gems with counts
            for gem in self.data_loader.GEM_COLS:
                if gem in cluster_data.columns:
                    count = cluster_data[gem].sum()
                    if count > 0:
                        profile['top_gems'].append((gem, count))
                        profile['gem_counts'][gem] = count
            profile['top_gems'].sort(key=lambda x: x[1], reverse=True)

            # Bands with counts
            for band in self.data_loader.BAND_COLS:
                if band in cluster_data.columns:
                    count = cluster_data[band].sum()
                    if count > 0:
                        profile['top_bands'].append((band, count))
                        profile['band_counts'][band] = count
            profile['top_bands'].sort(key=lambda x: x[1], reverse=True)

            # Dominant colors
            color_counts = Counter()
            for _, row in cluster_data.iterrows():
                colors = self.data_loader.get_item_colors(row)
                color_counts.update(colors)
            profile['dominant_colors'] = color_counts.most_common(7) 

            self.cluster_profiles[cluster_id] = profile

    def _calculate_style_score(self, row: pd.Series, styles: List[str]) -> float:
        if not styles:
            return 0.0

        target_gems, target_bands = set(), set()
        for style in styles:
            if style in self.STYLE_PROFILES:
                target_gems.update(self.STYLE_PROFILES[style]['gems'])
                target_bands.update(self.STYLE_PROFILES[style]['bands'])

        gem_matches = sum(1 for gem in target_gems if gem in row.index and row.get(gem, 0) == 1)
        band_matches = sum(1 for band in target_bands if band in row.index and row.get(band, 0) == 1)

        max_gems = len(target_gems) if target_gems else 1
        max_bands = len(target_bands) if target_bands else 1

        gem_prop = min(gem_matches / max_gems, 1.0)
        band_prop = min(band_matches / max_bands, 1.0)

        raw = gem_prop * self.STYLE_GEM_WEIGHT + band_prop * self.STYLE_BAND_WEIGHT
        
        if 'modern' in styles:
          if row.get('yellow gold', 0) == 1 or row.get('pink gold', 0) == 1:
              raw *= 0.6  # Heavy penalty for warm gold in a "Modern" search

        # --- NEW PURITY CHECK ---
        # If user wants Bohemian/Vintage but item is "Diamond-heavy" (Luxurious), 
        # apply a small penalty so pure Bohemian pieces rank higher.
        is_diamond_heavy = row.get('diamonds', 0) == 1 or row.get('diamond', 0) == 1
        if any(s in ['bohemian', 'vintage', 'minimalist'] for s in styles) and is_diamond_heavy:
            raw *= 0.8  # 20% penalty for diamond-heavy items in niche style searches
            
        return raw * 10 * self.WEIGHT_STYLE

    def _calculate_color_score(self, row: pd.Series, preferred_colors: List[str]) -> float:
        # Calculate normalized color score (max 3.0 points with 15% weight)
        if not preferred_colors:
            return 0.0
        item_colors = self.data_loader.get_item_colors(row)
        matches = len(set(item_colors) & set(preferred_colors))
        raw = min(matches / len(preferred_colors), 1.0)
        return raw * 10 * self.WEIGHT_COLOR

    def _calculate_cluster_style_bonus(self, row: pd.Series,
                                        preferred_styles: List[str],
                                        preferred_colors: List[str]) -> Tuple[float, dict]:
        """
        Calculate cluster bonus based on BOTH style and color alignment.

        This is the key improvement: items in clusters whose assigned styles
        match the user's preferred styles get a bonus.
        """
        cluster_id = row['cluster']
        cluster_style_info = self.cluster_styles[cluster_id]
        cluster_profile = self.cluster_profiles[cluster_id]

        # Style alignment bonus (50% of cluster bonus)
        style_bonus = 0.0
        if preferred_styles:
            # Check if any preferred style matches cluster's styles
            cluster_style_scores = cluster_style_info['all_styles']
            matching_scores = []
            for pref_style in preferred_styles:
                if pref_style in cluster_style_scores:
                    matching_scores.append(cluster_style_scores[pref_style])

            if matching_scores:
                # Average of matching style scores
                style_bonus = sum(matching_scores) / len(matching_scores)

        # Color alignment bonus (50% of cluster bonus)
        color_bonus = 0.0
        if preferred_colors:
            cluster_colors = [c[0] for c in cluster_profile['dominant_colors'][:3]]
            overlap = len(set(cluster_colors) & set(preferred_colors))
            max_overlap = min(len(cluster_colors), len(preferred_colors))
            color_bonus = overlap / max_overlap if max_overlap > 0 else 0

        # Combined bonus (equal weight to style and color alignment)
        raw_bonus = (style_bonus * 0.5 + color_bonus * 0.5)
        final_score = raw_bonus * 10 * self.WEIGHT_CLUSTER

        breakdown = {
            'cluster_id': cluster_id,
            'cluster_primary_style': cluster_style_info['primary_style'],
            'cluster_style_score': cluster_style_info['primary_score'],
            'style_alignment': round(style_bonus, 3),
            'color_alignment': round(color_bonus, 3),
            'raw_bonus': round(raw_bonus, 3)
        }

        return final_score, breakdown

    def recommend(self, preferred_colors=None, preferred_styles=None,
                  min_price=None, max_price=None, min_quantity=1, top_k=10,
                  return_breakdown=False):
        """Generate recommendations with cluster-style aware scoring."""
        preferred_colors = preferred_colors or []
        preferred_styles = preferred_styles or []

        # Hard filters
        candidates = self.data_loader.df[self.data_loader.df['quantity'] >= min_quantity].copy()
        if min_price:
            candidates = candidates[candidates['price'] >= min_price]
        if max_price:
            candidates = candidates[candidates['price'] <= max_price]

        if len(candidates) == 0:
            return pd.DataFrame()

        # Calculate scores
        scores = []
        for idx, row in candidates.iterrows():
            style_score = self._calculate_style_score(row, preferred_styles)
            color_score = self._calculate_color_score(row, preferred_colors)
            cluster_bonus, cluster_breakdown = self._calculate_cluster_style_bonus(
                row, preferred_styles, preferred_colors
            )

            total_score = style_score + color_score + cluster_bonus

            score_entry = {
                'index': idx,
                'style_score': round(style_score, 2),
                'color_score': round(color_score, 2),
                'cluster_bonus': round(cluster_bonus, 2),
                'total_score': round(total_score, 2),
                'cluster_style': cluster_breakdown['cluster_primary_style']
            }

            if return_breakdown:
                score_entry['cluster_breakdown'] = cluster_breakdown

            scores.append(score_entry)

        scores_df = pd.DataFrame(scores).sort_values('total_score', ascending=False)
        top_indices = scores_df.head(top_k)['index'].tolist()

        results = candidates.loc[top_indices].copy()
        results = results.merge(scores_df, left_index=True, right_on='index')

        cols = ['ref', 'title', 'price', 'image', 'quantity', 'cluster', 'cluster_style',
                'style_score', 'color_score', 'cluster_bonus', 'total_score']

        if return_breakdown:
            cols.append('cluster_breakdown')

        return results[[c for c in cols if c in results.columns]].reset_index(drop=True)

    def print_cluster_summary(self):
        #Print cluster analysis with style assignments
        print("CLUSTER ANALYSIS WITH STYLE ASSIGNMENTS\n")

        for cid in range(self.n_clusters):
            profile = self.cluster_profiles[cid]
            style_info = self.cluster_styles[cid]

            print(f"\n{'─'*70}")
            print(f"CLUSTER {cid}: {style_info['primary_style'].upper()}")
            print(f"{'─'*70}")
            print(f"  Items: {profile['size']} | Avg Price: ${profile['avg_price']:,.0f}")

            # Style assignments
            print(f"\n  Style Assignments:")
            for style, score in style_info['top_3_styles']:
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"     {style:<12} [{bar}] {score:.1%}")

            # Materials
            if profile['categories']:
                cats = ", ".join([f"{c[0]}({c[1]})" for c in profile['categories'][:3]])
                print(f"\n  Categories: {cats}")

            if profile['top_gems']:
                gems = ", ".join([g[0] for g in profile['top_gems'][:4]])
                print(f"  Top Gems: {gems}")

            if profile['top_bands']:
                bands = ", ".join([b[0] for b in profile['top_bands'][:3]])
                print(f"  Top Bands: {bands}")

            if profile['dominant_colors']:
                colors = ", ".join([c[0] for c in profile['dominant_colors'][:4]])
                print(f"  Colors: {colors}")

    def get_cluster_style_mapping(self) -> pd.DataFrame:
        """Return a DataFrame showing cluster-to-style mapping."""
        rows = []
        for cid in range(self.n_clusters):
            style_info = self.cluster_styles[cid]
            profile = self.cluster_profiles[cid]
            rows.append({
                'cluster': cid,
                'primary_style': style_info['primary_style'],
                'style_confidence': f"{style_info['primary_score']:.1%}",
                'secondary_styles': ', '.join([s[0] for s in style_info['top_3_styles'][1:3]]),
                'size': profile['size'],
                'avg_price': f"${profile['avg_price']:,.0f}"
            })
        return pd.DataFrame(rows)


def main():
    """Demo the improved system with cluster-style mapping."""

    print("Loading jewelry data...")
    loader = JewelryDataLoader('ready_for_ml_adjusted_if.csv')
    df = loader.load()
    print(f"Loaded {len(df)} jewelry items")

    print("\nTraining K-Modes clustering...")
    recommender = JewelryRecommender(loader, n_clusters=8)
    recommender.fit()

    # Show cluster-style mapping
    print("\n" + "="*70)
    print("CLUSTER → STYLE MAPPING")
    print("="*70)
    mapping_df = recommender.get_cluster_style_mapping()
    print(mapping_df.to_string(index=False))

    # Detailed cluster analysis
    recommender.print_cluster_summary()

    # Demo recommendations
    print("\n" + "="*70)
    print("RECOMMENDATION DEMO 1")
    print("="*70)

    print("\n--- Romantic Style, Pink Colors, $2000-$4000 ---")
    results1 = recommender.recommend(
        preferred_colors=['pink', 'red'],
        preferred_styles=['romantic'],
        min_price=2000,
        max_price=4000,
        top_k=5
    )
    display_cols = ['ref', 'title', 'price', 'cluster_style', 'total_score', 'image']
    pd.set_option('display.max_colwidth', None)  # Show full column width
    pd.set_option('display.width', None)  # Auto-detect terminal width
    print(results1[display_cols].to_string(index=False))

    print("\n" + "="*70)
    print("RECOMMENDATION DEMO 2")
    print("="*70)

    print("\n--- Bohemian Style, blue Colors, $500-$370000 ---")
    results2 = recommender.recommend(
        preferred_colors=['orange', 'blue'],
        preferred_styles=['bohemian', 'romantic'],
        min_price=500,
        max_price=370000,
        top_k=5
    )

    display_cols = ['ref', 'title', 'price', 'cluster_style', 'total_score', 'image']
    pd.set_option('display.max_colwidth', None)  # Show full column width
    pd.set_option('display.width', None)  # Auto-detect terminal width
    print(results2[display_cols].to_string(index=False))

    print("\n" + "="*70)
    print("RECOMMENDATION DEMO 3")
    print("="*70)

    print("\n--- Vintage Style, black Colors, $5000-$10000 ---")
    results2 = recommender.recommend(
        preferred_colors=['black', 'brown'],
        preferred_styles=['vintage'],
        min_price=5000,
        max_price=10000,
        top_k=5
    )

    display_cols = ['ref', 'title', 'price', 'cluster_style', 'total_score', 'image']
    pd.set_option('display.max_colwidth', None)  # Show full column width
    pd.set_option('display.width', None)  # Auto-detect terminal width
    print(results2[display_cols].to_string(index=False))

    print("\n" + "="*70)
    print("RECOMMENDATION DEMO 4")
    print("="*70)

    print("\n--- Bold Style, red Colors, $1000-$4000 ---")
    results2 = recommender.recommend(
        preferred_colors=['red', 'green'],
        preferred_styles=['bold'],
        min_price=1000,
        max_price=4000,
        top_k=5
    )

    display_cols = ['ref', 'title', 'price', 'cluster_style', 'total_score', 'image']
    pd.set_option('display.max_colwidth', None)  # Show full column width
    pd.set_option('display.width', None)  # Auto-detect terminal width
    print(results2[display_cols].to_string(index=False))

    print("\n" + "="*70)
    print("RECOMMENDATION DEMO 5")
    print("="*70)

    print("\n--- Classic Style, white Colors, $10000-$50000 ---")
    results2 = recommender.recommend(
        preferred_colors=['white', 'blue'],
        preferred_styles=['classic'],
        min_price=10000,
        max_price=50000,
        top_k=5
    )

    display_cols = ['ref', 'title', 'price', 'cluster_style', 'total_score', 'image']
    pd.set_option('display.max_colwidth', None)  # Show full column width
    pd.set_option('display.width', None)  # Auto-detect terminal width
    print(results2[display_cols].to_string(index=False))

    print("\n" + "="*70)
    print("RECOMMENDATION DEMO 6")
    print("="*70)

    print("\n--- Modern Style, green Colors, $500-$2000 ---")
    results2 = recommender.recommend(
        preferred_colors=['green', 'yellow'],
        preferred_styles=['modern'],
        min_price=500,
        max_price=2000,
        top_k=5
    )

    display_cols = ['ref', 'title', 'price', 'cluster_style', 'total_score', 'image']
    pd.set_option('display.max_colwidth', None)  # Show full column width
    pd.set_option('display.width', None)  # Auto-detect terminal width
    print(results2[display_cols].to_string(index=False))

    print("\n" + "="*70)
    print("RECOMMENDATION DEMO 7")
    print("="*70)

    print("\n--- Luxurious Style, white Colors, $100000-$370000 ---")
    results2 = recommender.recommend(
        preferred_colors=['red', 'white'],
        preferred_styles=['luxurious'],
        min_price=100000,
        max_price=370000,
        top_k=5
    )

    display_cols = ['ref', 'title', 'price', 'cluster_style', 'total_score', 'image']
    pd.set_option('display.max_colwidth', None)  # Show full column width
    pd.set_option('display.width', None)  # Auto-detect terminal width
    print(results2[display_cols].to_string(index=False))
    return recommender, loader

if __name__ == "__main__":
    recommender, loader = main()