"""
Export ML Model Results to Supabase Database
Uses in-memory cluster-style mapping from trained model
"""

import os
from supabase import create_client, Client
from jewelry_ml_model import JewelryDataLoader, JewelryRecommender
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase credentials
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')  # Use service key for admin access

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

print("="*70)
print("EXPORTING ML MODEL TO SUPABASE")
print("="*70)

# 1. Train ML Model
print("\n[1/4] Training ML Model in memory...")
loader = JewelryDataLoader('ready_for_ml_adjusted_if.csv')
df = loader.load()
print(f"Loaded {len(df)} items")

recommender = JewelryRecommender(loader, n_clusters=8)
recommender.fit()
print(f"Trained K-Modes clustering with 8 clusters")

# 2. Export Cluster Metadata using get_cluster_style_mapping()
print("\n[2/4] Exporting Cluster Metadata...")
mapping_df = recommender.get_cluster_style_mapping()

for _, row in mapping_df.iterrows():
    cluster_data = {
        'id': int(row['cluster']),
        'primary_style': row['primary_style'],
        'style_confidence': float(row['style_confidence'].strip('%')) / 100,
        'secondary_styles': row['secondary_styles'].split(', ') if row['secondary_styles'] else [],
        'avg_price': float(row['avg_price'].replace('$','').replace(',','')),
        'item_count': int(row['size']),
        'description': f"Cluster specializing in {row['primary_style']} jewelry",
        'updated_at': 'NOW()'
    }
    result = supabase.table('clusters').upsert(cluster_data).execute()
    print(f"Cluster {cluster_data['id']}: {cluster_data['primary_style']} ({cluster_data['item_count']} items)")

# 3. Export Item-Level Data with consistent cluster styles
print("\n[3/4] Exporting Item Data to jewelry_items...")
updated_count = 0
error_count = 0

for idx, row in df.iterrows():
    try:
        ref = str(row['ref']).strip()
        cluster_id = int(row['cluster'])
        cluster_style_info = recommender.cluster_styles[cluster_id]
        colors = loader.get_item_colors(row)

        # Determine item-level style tags
        item_style_scores = {}
        for style_name in recommender.STYLE_PROFILES.keys():
            score = recommender._calculate_style_score(row, [style_name])
            if score > 0:
                item_style_scores[style_name] = round(score, 3)

        if item_style_scores:
            sorted_styles = sorted(item_style_scores.items(), key=lambda x: x[1], reverse=True)
            dominant_style = sorted_styles[0][0]
            style_tags = [s[0] for s in sorted_styles[:3]]
        else:
            dominant_style = cluster_style_info['primary_style']
            style_tags = [dominant_style]

        update_data = {
            'cluster_id': cluster_id,
            'dominant_style': dominant_style,
            'style_tags': style_tags,
            'color_tags': colors,
            'style_confidence': float(cluster_style_info['primary_score'])
        }

        result = supabase.table('jewelry_items').update(update_data).eq('ref', ref).execute()

        if result.data:
            updated_count += 1
            if updated_count % 50 == 0:
                print(f"  Processed {updated_count} items...")
        else:
            error_count += 1
            print(f"Warning: ref '{ref}' not found in database")

    except Exception as e:
        error_count += 1
        print(f"Error processing ref '{ref}': {e}")

print(f"\nUpdated {updated_count} items")
if error_count > 0:
    print(f"{error_count} items had errors (likely not in database yet)")

# 4. Verify Export
print("\n[4/4] Verifying Export...")
result = supabase.table('jewelry_items').select(
    'id, ref, style_tags, color_tags, cluster_id'
).limit(5).execute()

print("\nSample of exported data:")
for item in result.data:
    print(f"   {item['ref']}: cluster={item['cluster_id']}, styles={item['style_tags']}, colors={item['color_tags']}")

print("\n" + "="*70)
print("EXPORT COMPLETE!")
print("="*70)
