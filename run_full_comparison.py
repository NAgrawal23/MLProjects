#!/usr/bin/env python3
"""
IMPROVED: View Similarity with FULL comparison (tables, columns, joins, filters)

This script demonstrates that the ViewSimilarityFinder DOES compare:
- Tables (50% weight)
- Columns (25% weight) 
- Join types (15% weight)
- Structure size (10% weight)

The composite similarity score includes ALL these factors.
"""

from view_similarity_finder_1 import ViewSimilarityFinder
import pandas as pd
import time
from datetime import datetime
import os

# =============================================================================
# LOGGING: Generate detailed analysis log file
# =============================================================================

def generate_analysis_log(finder, results, min_similarity, min_table_overlap, analysis_time, output_prefix='similarity_analysis'):
    """
    Generate a detailed log file with view information and similarity analysis
    
    Args:
        finder: ViewSimilarityFinder instance with loaded views
        results: DataFrame with similarity results
        min_similarity: Minimum similarity threshold used
        min_table_overlap: Minimum table overlap threshold used
        analysis_time: Total analysis time in seconds
        output_prefix: Prefix for log filename
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'{output_prefix}_{timestamp}.log'
    
    with open(log_file, 'w') as f:
        # Header
        f.write("="*100 + "\n")
        f.write("VIEW SIMILARITY ANALYSIS LOG\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analysis Duration: {analysis_time:.2f} seconds\n\n")
        
        # Configuration
        f.write("-"*100 + "\n")
        f.write("ANALYSIS CONFIGURATION\n")
        f.write("-"*100 + "\n")
        f.write(f"Minimum Table Overlap Threshold: {min_table_overlap*100:.0f}%\n")
        f.write(f"Minimum Composite Similarity Threshold: {min_similarity*100:.0f}%\n")
        f.write(f"Total Views Analyzed: {len(finder.index.view_names)}\n")
        f.write(f"Unique Tables Found: {len(finder.index.all_tables)}\n")
        f.write(f"Unique Columns Found: {len(finder.index.all_columns)}\n\n")
        
        # All views inventory
        f.write("-"*100 + "\n")
        f.write("ALL VIEWS INVENTORY\n")
        f.write("-"*100 + "\n")
        f.write(f"{'View Name':<50} {'Tables':<30} {'Table Count':<15}\n")
        f.write("-"*100 + "\n")
        
        for view_id, view_name in enumerate(finder.index.view_names):
            features = finder.index.view_features[view_id]
            tables = ', '.join(sorted(features['tables']))
            f.write(f"{view_name:<50} {tables:<30} {len(features['tables']):<15}\n")
        
        f.write("\n")
        
        # Summary statistics
        f.write("-"*100 + "\n")
        f.write("SIMILARITY ANALYSIS SUMMARY\n")
        f.write("-"*100 + "\n")
        
        if not results.empty:
            f.write(f"Total Similar Pairs Found: {len(results)}\n")
            f.write(f"Average Composite Similarity: {results['similarity_score'].mean():.2%}\n")
            f.write(f"Average Table Overlap: {results['table_overlap'].mean():.2%}\n")
            f.write(f"Average Column Overlap: {results['column_overlap'].mean():.2%}\n\n")
            
            f.write("Similarity Distribution:\n")
            f.write(f"  Exact matches (100%): {len(results[results['similarity_score'] >= 0.99])}\n")
            f.write(f"  Very high (80-99%): {len(results[(results['similarity_score'] >= 0.8) & (results['similarity_score'] < 0.99)])}\n")
            f.write(f"  High (60-80%): {len(results[(results['similarity_score'] >= 0.6) & (results['similarity_score'] < 0.8)])}\n")
            f.write(f"  Medium (40-60%): {len(results[(results['similarity_score'] >= 0.4) & (results['similarity_score'] < 0.6)])}\n")
            f.write(f"  Low (30-40%): {len(results[(results['similarity_score'] >= 0.3) & (results['similarity_score'] < 0.4)])}\n\n")
        else:
            f.write("No similar pairs found above threshold.\n\n")
        
        # Detailed similarity pairs analysis
        f.write("-"*100 + "\n")
        f.write("DETAILED SIMILARITY PAIRS ANALYSIS\n")
        f.write("-"*100 + "\n")
        
        if not results.empty:
            # Sort by similarity descending
            results_sorted = results.sort_values('similarity_score', ascending=False)
            
            for idx, (_, row) in enumerate(results_sorted.iterrows(), 1):
                f.write(f"\n[PAIR {idx}]\n")
                f.write(f"View 1: {row['source_view']}\n")
                f.write(f"View 2: {row['similar_view']}\n")
                f.write(f"Composite Similarity Score: {row['similarity_score']:.2%}\n")
                f.write(f"  ├─ Table Overlap: {row['table_overlap']:.2%}\n")
                f.write(f"  ├─ Column Overlap: {row['column_overlap']:.2%}\n")
                f.write(f"  ├─ Exact Match: {'YES' if row['is_exact_match'] else 'NO'}\n")
                f.write(f"  ├─ Table Count Difference: {row['table_count_diff']}\n")
                f.write(f"  └─ Column Count Difference: {row['column_count_diff']}\n")
                f.write(f"Common Tables: {row['common_tables']}\n")
                
                # Get view details from index
                view1_id = finder.index.view_ids_map.get(row['source_view'])
                view2_id = finder.index.view_ids_map.get(row['similar_view'])
                
                if view1_id is not None and view2_id is not None:
                    feat1 = finder.index.view_features[view1_id]
                    feat2 = finder.index.view_features[view2_id]
                    
                    only_in_view1 = feat1['tables'] - feat2['tables']
                    only_in_view2 = feat2['tables'] - feat1['tables']
                    
                    if only_in_view1:
                        f.write(f"Tables Only in {row['source_view']}: {', '.join(sorted(only_in_view1))}\n")
                    if only_in_view2:
                        f.write(f"Tables Only in {row['similar_view']}: {', '.join(sorted(only_in_view2))}\n")
                    
                    # Join info
                    joins1 = feat1['joins']
                    joins2 = feat2['joins']
                    f.write(f"Join Types in {row['source_view']}: {', '.join(joins1) if joins1 else 'None'}\n")
                    f.write(f"Join Types in {row['similar_view']}: {', '.join(joins2) if joins2 else 'None'}\n")
                
                f.write("-"*100 + "\n")
        else:
            f.write("No similar pairs to analyze.\n")
        
        # Load errors (if any)
        if finder.index.load_errors:
            f.write("\n" + "-"*100 + "\n")
            f.write("LOAD ERRORS/WARNINGS\n")
            f.write("-"*100 + "\n")
            for error in finder.index.load_errors:
                f.write(f"View: {error['view_name']}\n")
                f.write(f"Error: {error['error']}\n\n")
        
        # Footer
        f.write("\n" + "="*100 + "\n")
        f.write(f"Log generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n")
    
    return log_file


# =============================================================================

def diagnose_filtering(finder, min_table_overlap=0.3, min_similarity=0.3):
    """
    Show how many view pairs are filtered at each stage
    """
    diag_start = time.time()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC: Where are similar views being filtered?")
    print("="*80)
    
    n_views = len(finder.index.view_names)
    total_possible_pairs = (n_views * (n_views - 1)) // 2
    
    print(f"\nTotal views loaded: {n_views}")
    print(f"Total possible unique pairs: {total_possible_pairs:,}")
    
    # Count candidates at table_overlap threshold
    candidates_found = 0
    for view_id in range(n_views):
        source_features = finder.index.view_features[view_id]
        candidates = finder.index.find_candidates_by_tables(
            source_features['tables'], min_table_overlap
        )
        candidates_found += len([c for c in candidates if c > view_id])  # Count each pair once
    
    print(f"\nAfter table overlap filter ({min_table_overlap*100:.0f}%):")
    print(f"  Candidate pairs with shared tables: {candidates_found:,}")
    
    # Now count after similarity threshold
    results = finder.find_all_similarities(
        top_k=100,  # Get many to see how many pass similarity threshold
        min_similarity=0.0,  # No similarity threshold yet
        min_table_overlap=min_table_overlap,
        output_file=None
    )
    
    print(f"\nAfter computing similarities (no threshold yet):")
    print(f"  Total pairs compared: {len(results):,}")
    
    # Now filter by similarity
    above_threshold = len(results[results['similarity_score'] >= min_similarity])
    print(f"\nAfter similarity threshold ({min_similarity*100:.0f}%):")
    print(f"  Pairs above threshold: {above_threshold:,}")
    
    # Show distribution
    print(f"\nSimilarity score distribution:")
    print(f"  100% exact: {len(results[results['similarity_score'] >= 0.99]):,}")
    print(f"  80-99%: {len(results[(results['similarity_score'] >= 0.8) & (results['similarity_score'] < 0.99)]):,}")
    print(f"  60-80%: {len(results[(results['similarity_score'] >= 0.6) & (results['similarity_score'] < 0.8)]):,}")
    print(f"  40-60%: {len(results[(results['similarity_score'] >= 0.4) & (results['similarity_score'] < 0.6)]):,}")
    print(f"  20-40%: {len(results[(results['similarity_score'] >= 0.2) & (results['similarity_score'] < 0.4)]):,}")
    print(f"  0-20%: {len(results[results['similarity_score'] < 0.2]):,}")
    
    diag_time = time.time() - diag_start
    print(f"\n⏱️  Diagnostic completed in {diag_time:.2f} seconds")
    
    print(f"\n💡 RECOMMENDATION:")
    if above_threshold < 5:
        print(f"   Very few pairs found! Try:")
        print(f"   - Lowering min_table_overlap from {min_table_overlap*100:.0f}% to 0.1 or 0.15")
        print(f"   - Lowering min_similarity from {min_similarity*100:.0f}% to 0.1 or 0.2")
        print(f"   - Or your views truly don't share many tables/columns")
    elif above_threshold < 20:
        print(f"   Few pairs found. You might try lowering thresholds.")
    else:
        print(f"   Good number of similar pairs found!")


# =============================================================================
# CONNECT TO STARBURST AND LOAD VIEWS
# =============================================================================

def load_views_from_starburst(dsn, username, password, query):
    """
    Load views from Starburst with FULL similarity comparison
    """
    
    print("="*80)
    print("LOADING VIEWS FROM STARBURST")
    print("="*80)
    
    finder = ViewSimilarityFinder()
    
    # Connect
    if not finder.connect_to_starburst(dsn, username, password):
        print("Failed to connect!")
        return None
    
    # Load views
    try:
        finder.load_views_from_query(
            query,
            view_name_col='view_name',
            view_json_col='lineage',
            batch_size=100
        )
        
        print(f"\n✓ Successfully loaded {len(finder.index.view_names)} views")
        return finder
        
    except Exception as e:
        print(f"✗ Error loading views: {e}")
        return None


# =============================================================================
# FIND SIMILARITIES WITH FULL COMPARISON
# =============================================================================

def find_and_analyze_similarities(finder, min_similarity=0.3, min_table_overlap=0.3, output_file=None):
    """
    Find similar views using COMPOSITE similarity (tables + columns + joins + structure)
    
    The similarity_score in the output IS the composite score that includes:
    - 50% table overlap
    - 25% column overlap
    - 15% join pattern similarity
    - 10% structure size similarity
    
    Args:
        finder: ViewSimilarityFinder instance with loaded views
        min_similarity: Minimum composite similarity threshold (0.0-1.0)
        min_table_overlap: Minimum table overlap to consider as candidates (0.0-1.0)
        output_file: Output CSV filename (if None, generates timestamped filename)
    """
    
    analysis_start = time.time()
    
    # Generate timestamped filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'full_similarity_report_{timestamp}.csv'
    
    print("\n" + "="*80)
    print("COMPUTING SIMILARITIES (Tables + Columns + Joins + Structure)")
    print("="*80)
    print(f"\nFiltering parameters:")
    print(f"  Minimum table overlap for candidates: {min_table_overlap*100:.0f}%")
    print(f"  Minimum composite similarity: {min_similarity*100:.0f}%")
    print(f"  Output file: {output_file}")
    
    # This method DOES compare everything
    results = finder.find_all_similarities(
        top_k=10,
        min_similarity=min_similarity,
        min_table_overlap=min_table_overlap,
        output_file=output_file
    )
    
    if results.empty:
        print("\nNo similar views found above threshold")
        analysis_time = time.time() - analysis_start
        print(f"⏱️  Analysis completed in {analysis_time:.2f} seconds")
        return results
    
    print("\n" + "="*80)
    print("SIMILARITY REPORT")
    print("="*80)
    
    print(f"\nTotal similar pairs found: {len(results)}")
    print(f"\nBreakdown:")
    print(f"  Exact matches (similarity = 1.0): {len(results[results['similarity_score'] >= 0.99])}")
    print(f"  Very high (0.8-0.99): {len(results[(results['similarity_score'] >= 0.8) & (results['similarity_score'] < 0.99)])}")
    print(f"  High (0.6-0.8): {len(results[(results['similarity_score'] >= 0.6) & (results['similarity_score'] < 0.8)])}")
    print(f"  Medium (0.4-0.6): {len(results[(results['similarity_score'] >= 0.4) & (results['similarity_score'] < 0.6)])}")
    print(f"  Low (0.3-0.4): {len(results[(results['similarity_score'] >= 0.3) & (results['similarity_score'] < 0.4)])}")
    
    print(f"\nAverage scores:")
    print(f"  Composite similarity: {results['similarity_score'].mean():.2%}")
    print(f"  Table overlap: {results['table_overlap'].mean():.2%}")
    print(f"  Column overlap: {results['column_overlap'].mean():.2%}")
    
    # Show top 10 most similar
    print(f"\nTop 10 most similar view pairs:")
    print("-" * 80)
    top_10 = results.nlargest(10, 'similarity_score')
    
    for idx, row in top_10.iterrows():
        print(f"\n{row['source_view']} ↔ {row['similar_view']}")
        print(f"  Composite Score: {row['similarity_score']:.2%}")
        print(f"  Table overlap: {row['table_overlap']:.2%}")
        print(f"  Column overlap: {row['column_overlap']:.2%}")
        print(f"  Exact match: {'YES' if row['is_exact_match'] else 'NO'}")
        print(f"  Common tables: {row['common_tables']}")
    
    # Analyze cases where tables match but overall score differs
    print("\n" + "="*80)
    print("ANALYZING: Same Tables, Different Columns/Joins")
    print("="*80)
    
    same_tables_diff_score = results[
        (results['table_overlap'] >= 0.9) &  # Same tables
        (results['similarity_score'] < 0.9)   # But different overall
    ]
    
    if len(same_tables_diff_score) > 0:
        print(f"\nFound {len(same_tables_diff_score)} view pairs with same tables but different columns/joins:")
        print("-" * 80)
        
        for idx, row in same_tables_diff_score.head(10).iterrows():
            print(f"\n{row['source_view']} ↔ {row['similar_view']}")
            print(f"  Tables: {row['table_overlap']:.0%} match (basically same)")
            print(f"  Columns: {row['column_overlap']:.0%} match")
            print(f"  Overall: {row['similarity_score']:.0%} similar")
            print(f"  → Difference is due to: ", end="")
            if row['column_overlap'] < 0.5:
                print("different columns selected")
            elif row['table_count_diff'] > 2:
                print("different number of tables")
            else:
                print("different join patterns or filters")
    else:
        print("\nNo views with same tables but different scores found.")
        print("This means all views with same tables also have similar columns/joins.")
    
    analysis_time = time.time() - analysis_start
    print(f"\n⏱️  Analysis completed in {analysis_time:.2f} seconds")
    
    return results


# =============================================================================
# DETAILED ANALYSIS: Column-Level Comparison
# =============================================================================

def analyze_column_differences(finder, view1_name, view2_name):
    """
    Show detailed column-level differences between two views
    """
    
    print("\n" + "="*80)
    print(f"DETAILED COMPARISON: {view1_name} vs {view2_name}")
    print("="*80)
    
    # Get view IDs
    view1_id = finder.index.view_ids_map.get(view1_name)
    view2_id = finder.index.view_ids_map.get(view2_name)
    
    if view1_id is None or view2_id is None:
        print("One or both views not found")
        return
    
    # Get features
    feat1 = finder.index.view_features[view1_id]
    feat2 = finder.index.view_features[view2_id]
    
    # Compare tables
    print("\nTABLES:")
    tables1 = feat1['tables']
    tables2 = feat2['tables']
    common_tables = tables1 & tables2
    only_in_1 = tables1 - tables2
    only_in_2 = tables2 - tables1
    
    print(f"  Common tables ({len(common_tables)}): {', '.join(sorted(common_tables))}")
    if only_in_1:
        print(f"  Only in {view1_name}: {', '.join(sorted(only_in_1))}")
    if only_in_2:
        print(f"  Only in {view2_name}: {', '.join(sorted(only_in_2))}")
    
    table_overlap = len(common_tables) / len(tables1 | tables2) if (tables1 | tables2) else 0
    print(f"  Table overlap: {table_overlap:.0%}")
    
    # Compare columns
    print("\nCOLUMNS:")
    columns1 = feat1['columns']
    columns2 = feat2['columns']
    
    if '*' in columns1 or '*' in columns2:
        print("  One or both views use SELECT *")
        print(f"  {view1_name}: {'*' if '*' in columns1 else f'{len(columns1)} specific columns'}")
        print(f"  {view2_name}: {'*' if '*' in columns2 else f'{len(columns2)} specific columns'}")
    else:
        common_cols = columns1 & columns2
        only_in_1 = columns1 - columns2
        only_in_2 = columns2 - columns1
        
        print(f"  Common columns ({len(common_cols)}): {', '.join(sorted(list(common_cols)[:10]))}")
        if len(common_cols) > 10:
            print(f"    ... and {len(common_cols) - 10} more")
        
        if only_in_1:
            print(f"  Only in {view1_name} ({len(only_in_1)}): {', '.join(sorted(list(only_in_1)[:5]))}")
            if len(only_in_1) > 5:
                print(f"    ... and {len(only_in_1) - 5} more")
        
        if only_in_2:
            print(f"  Only in {view2_name} ({len(only_in_2)}): {', '.join(sorted(list(only_in_2)[:5]))}")
            if len(only_in_2) > 5:
                print(f"    ... and {len(only_in_2) - 5} more")
        
        col_overlap = len(common_cols) / len(columns1 | columns2) if (columns1 | columns2) else 0
        print(f"  Column overlap: {col_overlap:.0%}")
    
    # Compare joins
    print("\nJOINS:")
    joins1 = feat1['joins']
    joins2 = feat2['joins']
    
    print(f"  {view1_name}: {', '.join(joins1) if joins1 else 'None'}")
    print(f"  {view2_name}: {', '.join(joins2) if joins2 else 'None'}")
    
    if joins1 == joins2:
        print("  ✓ Join types match")
    else:
        print("  ✗ Join types differ")
    
    # Compute composite similarity
    print("\nCOMPOSITE SIMILARITY BREAKDOWN:")
    
    # Use the same logic as the engine
    if feat1['has_wildcard'] or feat2['has_wildcard']:
        col_sim = 0.5
    else:
        col_sim = len(columns1 & columns2) / len(columns1 | columns2) if (columns1 | columns2) else 0
    
    # Join similarity
    if not joins1 and not joins2:
        join_sim = 1.0
    elif not joins1 or not joins2:
        join_sim = 0.0
    else:
        from collections import Counter
        counter1 = Counter(joins1)
        counter2 = Counter(joins2)
        common = sum((counter1 & counter2).values())
        join_sim = common / max(len(joins1), len(joins2))
    
    # Size similarity
    size_sim = 1.0 - abs(feat1['table_count'] - feat2['table_count']) / max(
        feat1['table_count'], feat2['table_count']
    ) if (feat1['table_count'] > 0 and feat2['table_count'] > 0) else 0.0
    
    composite = 0.50 * table_overlap + 0.25 * col_sim + 0.15 * join_sim + 0.10 * size_sim
    
    print(f"  Table similarity:     {table_overlap:.2%} (weight: 50%) = {0.50 * table_overlap:.2%}")
    print(f"  Column similarity:    {col_sim:.2%} (weight: 25%) = {0.25 * col_sim:.2%}")
    print(f"  Join similarity:      {join_sim:.2%} (weight: 15%) = {0.15 * join_sim:.2%}")
    print(f"  Size similarity:      {size_sim:.2%} (weight: 10%) = {0.10 * size_sim:.2%}")
    print(f"  " + "-"*70)
    print(f"  COMPOSITE SCORE:      {composite:.2%}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function showing FULL similarity comparison
    """
    
    main_start = time.time()
    
    print("\n" + "="*80)
    print("VIEW SIMILARITY FINDER - FULL COMPARISON DEMO")
    print("Demonstrates that columns, joins, and structure ARE compared")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration - UPDATE THESE
    DSN = "your_starburst_dsn"
    USERNAME = "your_username"
    PASSWORD = "your_password"
    
    # Query - UPDATE THIS
    QUERY = """
        SELECT 
            view_name,
            view_definition_json as view_json
        FROM your_catalog.your_schema.views_table
        WHERE view_name IS NOT NULL
        LIMIT 3500
    """
    
    # Load views
    finder = load_views_from_starburst(DSN, USERNAME, PASSWORD, QUERY)
    
    if finder is None:
        print("\nFailed to load views. Exiting.")
        return
    
    # DIAGNOSTIC: Show where filtering is happening
    diagnose_filtering(
        finder,
        min_table_overlap=0.3,
        min_similarity=0.3
    )
    
    # Find similarities with FULL comparison
    results = find_and_analyze_similarities(
        finder,
        min_similarity=0.3,        # Adjust this to find more/fewer pairs
        min_table_overlap=0.3,     # Adjust this to find more candidate pairs
        output_file=None           # Auto-generate timestamped filename
    )
    
    analysis_duration = time.time() - main_start
    
    # Generate detailed log file
    log_file = generate_analysis_log(
        finder,
        results,
        min_similarity=0.3,
        min_table_overlap=0.3,
        analysis_time=analysis_duration,
        output_prefix='similarity_analysis'
    )
    print(f"\n📋 Log file generated: {log_file}")
    
    # If we have results, show detailed comparison for top pair
    if not results.empty:
        top_pair = results.iloc[0]
        analyze_column_differences(
            finder,
            top_pair['source_view'],
            top_pair['similar_view']
        )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: What's Being Compared?")
    print("="*80)
    print("""
The similarity_score in your results IS a composite score that includes:

1. TABLES (50% weight):
   - Jaccard similarity of tables used
   - If tables don't overlap, similarity = 0 (early exit)

2. COLUMNS (25% weight):
   - Jaccard similarity of column names
   - If view uses SELECT *, column similarity = 0.5 (neutral)
   - Otherwise, compares specific column names

3. JOIN PATTERNS (15% weight):
   - Compares join types (INNER, LEFT, RIGHT, OUTER, etc.)
   - Counts how many join types match between views

4. STRUCTURE SIZE (10% weight):
   - Compares number of tables, columns, joins
   - Views with similar complexity score higher

EXAMPLE:
  View A: customers + orders, 5 columns, INNER JOIN
  View B: customers + orders, 3 columns, LEFT JOIN
  
  Result:
    - Table overlap: 100% (same tables)
    - Column overlap: 60% (some columns differ)
    - Join similarity: 0% (INNER vs LEFT)
    - Size similarity: 80% (5 vs 3 tables)
    
    Composite = 0.5×100% + 0.25×60% + 0.15×0% + 0.1×80%
              = 50% + 15% + 0% + 8%
              = 73% similarity

So even with same tables, you get different scores based on columns/joins!
    """)
    
    print("\n" + "="*80)
    print("FILES CREATED:")
    print("="*80)
    # List CSV files in current directory
    csv_files = [f for f in os.listdir('.') if f.startswith('full_similarity_report_') and f.endswith('.csv')]
    if csv_files:
        csv_files.sort(reverse=True)  # Most recent first
        for csv_file in csv_files[:3]:  # Show last 3
            file_size = os.path.getsize(csv_file) / 1024  # Size in KB
            print(f"  📄 {csv_file} ({file_size:.1f} KB)")
    
    print("\nColumn descriptions:")
    print("  - similarity_score: Composite score (0.0-1.0)")
    print("  - table_overlap: Table Jaccard similarity")
    print("  - column_overlap: Column Jaccard similarity")
    print("  - is_exact_match: Whether structures are identical")
    print("  - common_tables: Tables present in both views")
    print("="*80)
    
    # Total time
    total_time = time.time() - main_start
    print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()