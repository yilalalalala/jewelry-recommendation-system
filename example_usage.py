"""
Example usage of the Jewelry Recommendation System
"""

import json
from src.recommendation_system import JewelryRecommendationSystem

def main():
    # Initialize system
    system = JewelryRecommendationSystem()

    # Load data
    system.load_data('data/ready_for_ml_adjusted.csv')

    # Example 1: Wedding event recommendation
    print("\n" + "="*60)
    print("EXAMPLE 1: Getting recommendations for a formal wedding")
    print("="*60)

    recommendations = system.recommend_jewelry(
        budget_min=2000,
        budget_max=3000,
        style_preferences=['elegant', 'formal', 'classic'],
        event='wedding',
        category='earrings',
        material_preferences=['diamond', 'pearl'],
        top_k=6
    )

    print(f"\nFound {len(recommendations)} recommendations:\n")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']}")
        print(f"   Ref: {rec['ref']}")
        print(f"   Price: ${rec['price']:.2f}")
        print(f"   Score: {rec['score']:.2f}")
        print(f"   Styles: {', '.join(rec['styles'][:3])}")
        print(f"   Materials: {', '.join(rec['materials'][:3]) if rec['materials'] else 'N/A'}")
        print(f"   Explanation: {rec['explanation']}")
        print(f"   Stock: {rec['quantity']} available")
        print()

    # Example 2: Casual event recommendation
    print("\n" + "="*60)
    print("EXAMPLE 2: Casual brunch outfit")
    print("="*60)

    recommendations = system.recommend_jewelry(
        budget_min=500,
        budget_max=1000,
        style_preferences=['casual', 'minimalist'],
        event='casual_brunch',
        category='bracelets',
        top_k=5
    )

    print(f"\nFound {len(recommendations)} recommendations:\n")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} - ${rec['price']:.2f} (Score: {rec['score']:.2f})")

    # Show analytics
    print("\n" + "="*60)
    print("ANALYTICS DASHBOARD")
    print("="*60)

    trends = system.analyze_trends()
    print(json.dumps(trends, indent=2))

if __name__ == "__main__":
    main()
