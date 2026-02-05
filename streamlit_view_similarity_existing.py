#!/usr/bin/env python3
"""
streamlit_view_similarity_existing.py
=====================================
Streamlit UI for View Similarity Finder (uses view_similarity_finder_1.py)

This version works with your EXISTING code without requiring the NetworkX graph engine.

Features:
    1. DB Connection Manager
    2. Find Similar Views (single view lookup)
    3. Find Views by Tables (table-based search)
    4. All Similarities Report (complete dataset)

Run with:
    streamlit run streamlit_view_similarity_existing.py
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# Import the existing finder
from view_similarity_finder_1 import ViewSimilarityFinder


# ═══════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════

def init_session_state():
    """Initialize session state variables"""
    if 'finder' not in st.session_state:
        st.session_state.finder = None
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'views_loaded' not in st.session_state:
        st.session_state.views_loaded = False
    if 'connection_params' not in st.session_state:
        st.session_state.connection_params = {
            'dsn': '',
            'username': '',
            'password': '',
            'query': ''
        }
    if 'all_similarities' not in st.session_state:
        st.session_state.all_similarities = None


# ═══════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="View Similarity Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()


# ═══════════════════════════════════════════════════════════
# SIDEBAR - NAVIGATION
# ═══════════════════════════════════════════════════════════

st.sidebar.title("🔍 View Similarity Analyzer")
st.sidebar.markdown("---")

# Status indicators
if st.session_state.connected:
    st.sidebar.success("✅ Connected to Database")
else:
    st.sidebar.warning("⚠️ Not Connected")

if st.session_state.views_loaded:
    n_views = len(st.session_state.finder.index.view_names) if st.session_state.finder else 0
    st.sidebar.info(f"📊 {n_views} views loaded")
else:
    st.sidebar.info("📊 No views loaded")

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Select Feature",
    [
        "1️⃣ Database Connection",
        "2️⃣ Find Similar Views",
        "3️⃣ Find Views by Tables",
        "4️⃣ All Similarities Report"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This tool uses:
- **Set-based Indexing** for fast candidate retrieval
- **Composite Similarity**: Tables (50%), Columns (25%), Joins (15%), Size (10%)
- **Structure Hashing** for exact duplicate detection
""")


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def format_similarity_score(score: float) -> str:
    """Format score with color coding"""
    if score >= 0.9:
        return f"🟢 {score:.1%}"
    elif score >= 0.7:
        return f"🟡 {score:.1%}"
    elif score >= 0.5:
        return f"🟠 {score:.1%}"
    else:
        return f"🔴 {score:.1%}"


def create_similarity_gauge(score: float, title: str = "Similarity Score"):
    """Create a gauge chart for similarity score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 70, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': "%"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 90], 'color': "orange"},
                {'range': [90, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_similarity_distribution(df: pd.DataFrame):
    """Create distribution chart of similarity scores"""
    if df.empty:
        return None
    
    fig = px.histogram(
        df,
        x='similarity_score',
        nbins=20,
        title='Similarity Score Distribution',
        labels={'similarity_score': 'Similarity Score', 'count': 'Number of Pairs'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(
        xaxis_tickformat='.0%',
        height=300,
        showlegend=False
    )
    return fig


# ═══════════════════════════════════════════════════════════
# PAGE 1: DATABASE CONNECTION
# ═══════════════════════════════════════════════════════════

if page == "1️⃣ Database Connection":
    st.title("🔌 Database Connection")
    st.markdown("Connect to Starburst and load view definitions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Connection Settings")
        
        with st.form("connection_form"):
            dsn = st.text_input(
                "DSN Name",
                value=st.session_state.connection_params['dsn'],
                help="The ODBC Data Source Name configured on your system"
            )
            
            col_user, col_pass = st.columns(2)
            with col_user:
                username = st.text_input(
                    "Username (optional)",
                    value=st.session_state.connection_params['username'],
                    help="Leave empty if DSN handles authentication"
                )
            with col_pass:
                password = st.text_input(
                    "Password (optional)",
                    type="password",
                    value=st.session_state.connection_params['password'],
                    help="Leave empty if DSN handles authentication"
                )
            
            query = st.text_area(
                "SQL Query to Load Views",
                value=st.session_state.connection_params['query'] or """SELECT 
    view_name,
    view_definition_json as view_json
FROM your_catalog.your_schema.views_table
WHERE view_name IS NOT NULL
LIMIT 4000""",
                height=150,
                help="Query must return columns: view_name, view_json"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                connect_btn = st.form_submit_button("🔌 Connect & Load Views", type="primary", use_container_width=True)
            with col_btn2:
                disconnect_btn = st.form_submit_button("❌ Disconnect", use_container_width=True)
        
        if connect_btn:
            # Save parameters
            st.session_state.connection_params = {
                'dsn': dsn,
                'username': username,
                'password': password,
                'query': query
            }
            
            with st.spinner("Connecting to database and loading views..."):
                try:
                    # Create new finder instance
                    finder = ViewSimilarityFinder()
                    
                    # Connect
                    if finder.connect_to_starburst(dsn, username or None, password or None):
                        st.session_state.connected = True
                        
                        # Load views
                        finder.load_views_from_query(query)
                        
                        # Store in session
                        st.session_state.finder = finder
                        st.session_state.views_loaded = True
                        
                        st.success(f"✅ Successfully loaded {len(finder.index.view_names)} views!")
                        st.balloons()
                        
                        # Show summary
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        with col_s1:
                            st.metric("Total Views", len(finder.index.view_names))
                        with col_s2:
                            st.metric("Unique Tables", len(finder.index.all_tables))
                        with col_s3:
                            st.metric("Unique Columns", len(finder.index.all_columns))
                        with col_s4:
                            st.metric("Unique Structures", len(finder.index.structure_hash_index))
                        
                        time.sleep(1)
                        st.rerun()
                        
                    else:
                        st.error("❌ Failed to connect to database")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
        
        if disconnect_btn:
            if st.session_state.finder:
                st.session_state.finder.close()
            st.session_state.finder = None
            st.session_state.connected = False
            st.session_state.views_loaded = False
            st.session_state.all_similarities = None
            st.success("✅ Disconnected")
            time.sleep(1)
            st.rerun()
    
    with col2:
        st.subheader("Current Status")
        
        if st.session_state.views_loaded and st.session_state.finder:
            finder = st.session_state.finder
            
            st.info(f"""
            **Connection Status**: ✅ Connected  
            **Views Loaded**: {len(finder.index.view_names)}  
            **Unique Tables**: {len(finder.index.all_tables)}  
            **Unique Columns**: {len(finder.index.all_columns)}
            """)
            
            # Show errors if any
            if finder.index.load_errors:
                with st.expander(f"⚠️ {len(finder.index.load_errors)} Load Errors", expanded=False):
                    errors_df = pd.DataFrame(finder.index.load_errors)
                    st.dataframe(errors_df, use_container_width=True)
            
            # Show sample views
            with st.expander("📋 Sample Views (first 20)", expanded=False):
                sample_data = []
                for i in range(min(20, len(finder.index.view_names))):
                    feat = finder.index.view_features[i]
                    sample_data.append({
                        'View Name': finder.index.view_names[i],
                        'Tables': ', '.join(sorted(feat['tables'])),
                        'Table Count': feat['table_count']
                    })
                sample = pd.DataFrame(sample_data)
                st.dataframe(sample, use_container_width=True)
        
        else:
            st.warning("⚠️ Not connected. Fill in the form and click 'Connect & Load Views'")


# ═══════════════════════════════════════════════════════════
# PAGE 2: FIND SIMILAR VIEWS (Single View Lookup)
# ═══════════════════════════════════════════════════════════

elif page == "2️⃣ Find Similar Views":
    st.title("🔎 Find Similar Views")
    st.markdown("Search for views similar to a specific view")
    
    if not st.session_state.views_loaded:
        st.warning("⚠️ Please connect to database and load views first (Tab 1)")
        st.stop()
    
    finder = st.session_state.finder
    
    # Search form
    col1, col2 = st.columns([3, 1])
    
    with col1:
        view_name = st.selectbox(
            "Select View to Analyze",
            options=sorted(finder.index.view_names),
            help="Choose a view to find similar views"
        )
    
    with col2:
        top_k = st.number_input(
            "Top K Results",
            min_value=1,
            max_value=100,
            value=10,
            help="Number of similar views to return"
        )
    
    min_similarity = st.slider(
        "Minimum Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        format="%.0%",
        help="Only show views with similarity above this threshold"
    )
    
    if st.button("🔍 Search Similar Views", type="primary"):
        with st.spinner("Searching..."):
            results = finder.find_similar_views(view_name, top_k=top_k, min_similarity=min_similarity)
            
            if results.empty:
                st.warning(f"No similar views found for '{view_name}' above {min_similarity:.0%} threshold")
            else:
                st.success(f"Found {len(results)} similar views")
                
                # Show source view details
                st.subheader(f"📊 Source View: {view_name}")
                view_id = finder.index.view_ids_map[view_name]
                feat = finder.index.view_features[view_id]
                
                col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                with col_d1:
                    st.metric("Tables", feat['table_count'])
                with col_d2:
                    st.metric("Columns", feat['column_count'] if not feat['has_wildcard'] else "* (wildcard)")
                with col_d3:
                    st.metric("Joins", feat['join_count'])
                with col_d4:
                    st.metric("Has Wildcard", "Yes" if feat['has_wildcard'] else "No")
                
                st.markdown(f"**Tables**: {', '.join(sorted(feat['tables']))}")
                if not feat['has_wildcard']:
                    st.markdown(f"**Columns**: {', '.join(sorted(feat['columns'])[:20])}")
                if feat['joins']:
                    st.markdown(f"**Joins**: {', '.join(feat['joins'])}")
                
                st.markdown("---")
                
                # Results
                st.subheader(f"✨ Similar Views ({len(results)} found)")
                
                # Add formatted score column
                results_display = results.copy()
                results_display['Score'] = results_display['similarity_score'].apply(format_similarity_score)
                
                # Reorder columns for display
                display_cols = ['Score', 'similar_view', 'table_overlap', 'column_overlap', 'is_exact_match', 'common_tables']
                results_display = results_display[display_cols]
                results_display.columns = ['Score', 'View Name', 'Table Overlap', 'Column Overlap', 'Exact Match', 'Common Tables']
                
                st.dataframe(
                    results_display,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Detailed comparison for top result
                if len(results) > 0:
                    st.markdown("---")
                    st.subheader("🔬 Detailed Comparison: Top Match")
                    
                    top_match = results.iloc[0]
                    
                    col_c1, col_c2 = st.columns(2)
                    
                    with col_c1:
                        st.markdown(f"**Similarity Gauge**")
                        fig = create_similarity_gauge(top_match['similarity_score'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_c2:
                        st.markdown(f"**Breakdown**")
                        breakdown_data = {
                            'Metric': ['Tables', 'Columns', 'Overall'],
                            'Score': [
                                f"{top_match['table_overlap']:.1%}",
                                f"{top_match['column_overlap']:.1%}",
                                f"{top_match['similarity_score']:.1%}"
                            ]
                        }
                        st.dataframe(pd.DataFrame(breakdown_data), hide_index=True, use_container_width=True)
                    
                    # Side-by-side comparison
                    st.markdown("**Side-by-Side Comparison**")
                    comp_col1, comp_col2 = st.columns(2)
                    
                    match_id = finder.index.view_ids_map[top_match['similar_view']]
                    match_feat = finder.index.view_features[match_id]
                    
                    with comp_col1:
                        st.info(f"**{view_name}**")
                        st.markdown(f"Tables: `{', '.join(sorted(feat['tables']))}`")
                        if not feat['has_wildcard']:
                            st.markdown(f"Columns: `{', '.join(sorted(list(feat['columns'])[:10]))}`")
                    
                    with comp_col2:
                        st.info(f"**{top_match['similar_view']}**")
                        st.markdown(f"Tables: `{', '.join(sorted(match_feat['tables']))}`")
                        if not match_feat['has_wildcard']:
                            st.markdown(f"Columns: `{', '.join(sorted(list(match_feat['columns'])[:10]))}`")
                
                # Download button
                st.markdown("---")
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"similar_to_{view_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


# ═══════════════════════════════════════════════════════════
# PAGE 3: FIND VIEWS BY TABLES (Table-based Search)
# ═══════════════════════════════════════════════════════════

elif page == "3️⃣ Find Views by Tables":
    st.title("🔍 Find Views by Tables")
    st.markdown("Search for views that use specific tables")
    
    if not st.session_state.views_loaded:
        st.warning("⚠️ Please connect to database and load views first (Tab 1)")
        st.stop()
    
    finder = st.session_state.finder
    
    # Get all unique tables
    all_tables = sorted(finder.index.all_tables)
    
    st.info(f"📊 Total unique tables in dataset: {len(all_tables)}")
    
    # Table selection
    selected_tables = st.multiselect(
        "Select Tables to Search For",
        options=all_tables,
        help="Select one or more tables. Views containing ANY of these tables will be shown."
    )
    
    if not selected_tables:
        st.warning("⚠️ Please select at least one table to search")
        st.stop()
    
    # Search options
    col1, col2 = st.columns(2)
    with col1:
        match_mode = st.radio(
            "Match Mode",
            ["ANY (views with at least one of these tables)",
             "ALL (views with all of these tables)",
             "EXACT (views with exactly these tables)"],
            help="How to match the selected tables"
        )
    
    with col2:
        min_table_overlap = st.slider(
            "Minimum Table Overlap (for ANY mode)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            format="%.0%",
            disabled=match_mode != "ANY (views with at least one of these tables)"
        )
    
    if st.button("🔍 Search Views", type="primary"):
        selected_set = set(selected_tables)
        matched_views = []
        
        with st.spinner("Searching views..."):
            for vid in range(len(finder.index.view_names)):
                view_name = finder.index.view_names[vid]
                feat = finder.index.view_features[vid]
                view_tables = feat['tables']
                overlap = view_tables & selected_set
                
                matched = False
                if match_mode.startswith("ANY"):
                    # ANY: at least min_table_overlap
                    if overlap:
                        jaccard = len(overlap) / len(view_tables | selected_set)
                        if jaccard >= min_table_overlap:
                            matched = True
                            match_type = "Partial Match"
                            overlap_pct = jaccard
                elif match_mode.startswith("ALL"):
                    # ALL: contains all selected tables
                    if selected_set.issubset(view_tables):
                        matched = True
                        match_type = "Contains All"
                        overlap_pct = len(overlap) / len(view_tables)
                elif match_mode.startswith("EXACT"):
                    # EXACT: exactly these tables, no more, no less
                    if view_tables == selected_set:
                        matched = True
                        match_type = "Exact Match"
                        overlap_pct = 1.0
                
                if matched:
                    matched_views.append({
                        'View Name': view_name,
                        'Match Type': match_type,
                        'Overlap %': overlap_pct,
                        'Table Count': feat['table_count'],
                        'View Tables': ', '.join(sorted(view_tables)),
                        'Matched Tables': ', '.join(sorted(overlap)),
                        'Additional Tables': ', '.join(sorted(view_tables - selected_set)) if view_tables - selected_set else 'None'
                    })
        
        if not matched_views:
            st.warning(f"No views found matching the criteria")
        else:
            st.success(f"Found {len(matched_views)} views matching your criteria")
            
            results_df = pd.DataFrame(matched_views)
            results_df = results_df.sort_values('Overlap %', ascending=False)
            
            # Summary metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Total Matches", len(matched_views))
            with col_m2:
                exact_matches = len([v for v in matched_views if v['Match Type'] == 'Exact Match'])
                st.metric("Exact Matches", exact_matches)
            with col_m3:
                avg_overlap = results_df['Overlap %'].mean()
                st.metric("Avg Overlap", f"{avg_overlap:.1%}")
            
            st.markdown("---")
            
            # Display results
            st.subheader("📋 Matching Views")
            
            # Format for display
            display_df = results_df.copy()
            display_df['Overlap %'] = display_df['Overlap %'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Distribution chart
            if len(matched_views) > 1:
                st.markdown("---")
                st.subheader("📊 Distribution by Table Count")
                
                fig = px.histogram(
                    results_df,
                    x='Table Count',
                    title='Distribution of Views by Table Count',
                    nbins=20,
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Download
            st.markdown("---")
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"views_with_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


# ═══════════════════════════════════════════════════════════
# PAGE 4: ALL SIMILARITIES REPORT (Complete Dataset)
# ═══════════════════════════════════════════════════════════

elif page == "4️⃣ All Similarities Report":
    st.title("📊 All Similarities Report")
    st.markdown("Complete view similarity analysis across the entire dataset")
    
    if not st.session_state.views_loaded:
        st.warning("⚠️ Please connect to database and load views first (Tab 1)")
        st.stop()
    
    finder = st.session_state.finder
    
    # Filter controls
    st.subheader("🎛️ Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_sim_filter = st.slider(
            "Minimum Similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            format="%.0%"
        )
    
    with col2:
        show_exact_only = st.checkbox("Show Exact Matches Only", value=False)
    
    with col3:
        max_rows = st.number_input(
            "Max Rows to Display",
            min_value=10,
            max_value=10000,
            value=1000,
            step=100
        )
    
    # Compute or retrieve all similarities
    if st.session_state.all_similarities is not None:
        all_sims = st.session_state.all_similarities
    else:
        if st.button("🔄 Compute All Similarities", type="primary"):
            with st.spinner("Computing all view similarities... This may take a few minutes."):
                all_sims = finder.find_all_similarities(
                    top_k=100,
                    min_similarity=0.0,
                    min_table_overlap=0.3
                )
                st.session_state.all_similarities = all_sims
                st.success(f"✅ Computed {len(all_sims)} similarity pairs")
                st.rerun()
        else:
            st.info("👆 Click 'Compute All Similarities' to generate the complete similarity report")
            st.stop()
    
    # Apply filters
    filtered = all_sims[all_sims['similarity_score'] >= min_sim_filter]
    if show_exact_only:
        filtered = filtered[filtered['is_exact_match'] == True]
    
    filtered = filtered.head(max_rows)
    
    # Summary Statistics
    st.markdown("---")
    st.subheader("📈 Summary Statistics")
    
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    
    with col_s1:
        st.metric("Total Pairs (filtered)", len(filtered))
    with col_s2:
        st.metric("Total Pairs (all)", len(all_sims))
    with col_s3:
        if len(filtered) > 0:
            st.metric("Avg Similarity", f"{filtered['similarity_score'].mean():.1%}")
        else:
            st.metric("Avg Similarity", "N/A")
    with col_s4:
        exact_count = len(all_sims[all_sims['is_exact_match'] == True])
        st.metric("Exact Duplicates", exact_count)
    with col_s5:
        if len(filtered) > 0:
            st.metric("Max Similarity", f"{filtered['similarity_score'].max():.1%}")
        else:
            st.metric("Max Similarity", "N/A")
    
    # Distribution charts
    if len(filtered) > 0:
        st.markdown("---")
        st.subheader("📊 Distribution Analysis")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            fig1 = create_similarity_distribution(filtered)
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
        
        with col_c2:
            # Breakdown by similarity ranges
            bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
            labels = ['Low (0-30%)', 'Medium (30-50%)', 'High (50-70%)', 'Very High (70-90%)', 'Exact (90-100%)']
            filtered_copy = filtered.copy()
            filtered_copy['Range'] = pd.cut(filtered_copy['similarity_score'], bins=bins, labels=labels, include_lowest=True)
            range_counts = filtered_copy['Range'].value_counts().sort_index()
            
            fig2 = px.pie(
                values=range_counts.values,
                names=range_counts.index,
                title='Similarity Score Ranges',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Main data table
    st.markdown("---")
    st.subheader("📋 All Similar Pairs (Filtered)")
    
    if len(filtered) > 0:
        # Format for display
        display = filtered.copy()
        display['similarity_score'] = display['similarity_score'].apply(lambda x: f"{x:.1%}")
        display['table_overlap'] = display['table_overlap'].apply(lambda x: f"{x:.1%}")
        display['column_overlap'] = display['column_overlap'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )
        
        # Download options
        st.markdown("---")
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Results (CSV)",
                data=csv,
                file_name=f"filtered_similarities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_d2:
            csv_all = all_sims.to_csv(index=False)
            st.download_button(
                label="📥 Download All Results (CSV)",
                data=csv_all,
                file_name=f"all_similarities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.warning("No pairs match the current filters")


# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    View Similarity Analyzer v1.0<br>
    Using your existing view_similarity_finder_1.py
</div>
""", unsafe_allow_html=True)