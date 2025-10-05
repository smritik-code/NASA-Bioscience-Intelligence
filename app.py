import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import json
import base64
import requests
import re
import os
import hashlib
from datetime import datetime
import random

# Add audio dependencies
from gtts import gTTS
import io
import base64

st.set_page_config(
    page_title="NASA Bioscience Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE = "http://localhost:8000"

@st.cache_data
def load_precomputed_data():
    data_dir = "nasa_precomputed_data"
    data = {}

    files = ['publications.json', 'global_themes_network.json', 'search_index.json', 'metadata.json']

    for file in files:
        try:
            with open(f"{data_dir}/{file}", "r") as f:
                data[file.replace('.json', '')] = json.load(f)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")

    return data

precomputed = load_precomputed_data()

def text_to_speech_api(text):
    """Convert text to speech using backend API"""
    try:
        response = requests.post(
            f"{API_BASE}/api/audio/generate",
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get('audio_content')
    except Exception as e:
        st.error(f"Audio generation failed: {e}")
    return None

def generate_unique_key(base_key, *args):
    """Generate a unique key by hashing additional arguments"""
    key_string = base_key + "_" + "_".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()[:16]

def generate_mission_insights(pub_data):
    """Generate actionable mission planning insights"""
    themes = pub_data.get('theme_analysis', {}).get('theme_counts', {})
    tech_terms = pub_data.get('technical_terms', [])

    insights = []

    # Radiation insights
    if themes.get('radiation', 0) > 2:
        insights.append("Radiation Protection: Informs spacecraft shielding requirements and astronaut safety protocols")

    # Plant biology insights
    if themes.get('plant_biology', 0) > 2:
        insights.append("Food Production: Supports bioregenerative life support systems for long-duration missions")

    # Human physiology insights
    if themes.get('human_physiology', 0) > 2:
        insights.append("Crew Health: Guides exercise countermeasures and medical monitoring systems")

    # Microgravity insights
    if themes.get('microgravity', 0) > 2:
        insights.append("Gravity Effects: Critical for understanding physiological changes in space")

    # Microbiology insights
    if themes.get('microbiology', 0) > 2:
        insights.append("Microbiome: Essential for managing spacecraft hygiene and crew health")

    # Life support insights
    if themes.get('life_support', 0) > 2:
        insights.append("Life Support: Informs closed-loop system design for Moon/Mars habitats")

    # Technology insights
    if themes.get('technology', 0) > 2 or any(term in str(tech_terms).lower() for term in ['sensor', 'monitor', 'device', 'system']):
        insights.append("Technology: Hardware requirements for biological experiments in space")

    # Behavioral insights
    if themes.get('behavioral', 0) > 2:
        insights.append("Behavioral Health: Supports crew selection and psychological support systems")

    return insights if insights else ["General Research: Contributes to baseline space biology knowledge for future exploration"]

def calculate_theme_maturity():
    """Calculate research maturity scores for each theme"""
    themes_maturity = {}
    theme_publications = precomputed['global_themes_network']['theme_publications']

    for theme in precomputed['global_themes_network']['nodes']:
        theme_id = theme['id']
        pubs_count = len(theme_publications.get(theme_id, []))
        if pubs_count <= 200:
            maturity_score = min(pubs_count/4, 40) 
        elif pubs_count <= 260:
            maturity_score = 40 + min((pubs_count - 10)/2, 30) 
        else:
            maturity_score = 70 + min((pubs_count - 25)/ 1.5, 30)  
        
        themes_maturity[theme_id] = {
            'score': maturity_score,
            'publications': pubs_count
        }

    return themes_maturity


def generate_audio(text):
    try:

        text = text[:500]

        tts = gTTS(text=text, lang='en', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        return audio_base64
    except Exception as e:
        st.error(f"Audio generation failed: {e}")
        return None


def show_demo_scenarios():
    """Pre-built demo scenarios for competition judging"""
    st.header("Demo Scenarios")
    st.markdown("""
    Try these pre-built scenarios to experience the platform's capabilities for different user roles:
    """)

    scenario = st.selectbox(
        "Choose a scenario:",
        [
            "Select a scenario...",
            "Mission Planning: Radiation Protection",
            "Research: Plant Growth in Space",
            "Technology: Life Support Systems",
            "Crew Health: Long-duration Missions"
        ]
    )

    if scenario == "Mission Planning: Radiation Protection":
        st.success("Scenario: Designing radiation protection for Mars mission crew safety")
        st.info("User: Mission Architect | Goal: Identify radiation protection strategies from existing research")

        results = precomputed['search_index']['by_theme'].get('radiation', [])[:8]
        if results:
            st.subheader("Relevant Radiation Studies")
            for i, pmc_id in enumerate(results):
                pub_data = precomputed['publications'][pmc_id]
                with st.expander(f"Study: {pmc_id}"):
                    st.write("Summary:", pub_data.get('abstract_summary', 'No summary available'))
                    insights = generate_mission_insights(pub_data)
                    for insight in insights:
                        if "Radiation" in insight:
                            st.success(insight)

    elif scenario == "Research: Plant Growth in Space":
        st.success("Scenario: Developing sustainable food production for long missions")
        st.info("User: Research Scientist | Goal: Find plant biology research for space agriculture")

        results = precomputed['search_index']['by_theme'].get('plant_biology', [])[:8]
        if results:
            st.subheader("Plant Biology Research")
            for i, pmc_id in enumerate(results):
                pub_data = precomputed['publications'][pmc_id]
                with st.expander(f"Study: {pmc_id}"):
                    st.write("Summary:", pub_data.get('abstract_summary', 'No summary available'))
                    if st.button("Analyze Technical Concepts", key=f"plant_{pmc_id}"):
                        st.session_state.selected_publication = pmc_id
                        st.session_state.current_section = "Publication Analysis"
                        st.rerun()

    elif scenario == "Technology: Life Support Systems":
        st.success("Scenario: Engineering reliable life support for lunar habitats")
        st.info("User: Systems Engineer | Goal: Identify technology requirements from biological research")

        # Search for technology and life support terms
        tech_results = precomputed['search_index']['by_theme'].get('technology', [])
        life_results = precomputed['search_index']['by_theme'].get('life_support', [])
        combined_results = list(set(tech_results + life_results))[:8]

        if combined_results:
            st.subheader("Technology & Life Support Research")
            for pmc_id in combined_results:
                pub_data = precomputed['publications'][pmc_id]
                with st.expander(f"Study: {pmc_id}"):
                    st.write("Summary:", pub_data.get('abstract_summary', 'No summary available'))
                    insights = generate_mission_insights(pub_data)
                    for insight in insights:
                        st.info(insight)

    elif scenario != "Select a scenario...":
        st.info(f"Scenario '{scenario}' implementation in progress...")
        st.button("Explore Manually in Search Section")

def show_impact_dashboard():
    """Show research impact metrics"""
    st.header("Research Impact Dashboard")

    themes_maturity = calculate_theme_maturity()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_mission_studies = sum(1 for data in precomputed['publications'].values()
                                  if any(theme in data.get('theme_analysis', {}).get('theme_counts', {})
                                       for theme in ['radiation', 'plant_biology', 'human_physiology', 'life_support']))
        st.metric("Mission-Critical Studies", total_mission_studies)

    with col2:
        avg_themes_per_study = sum(len(data.get('theme_analysis', {}).get('theme_counts', {}))
                                 for data in precomputed['publications'].values()) / len(precomputed['publications'])
        st.metric("Avg Themes per Study", f"{avg_themes_per_study:.1f}")

    with col3:
        total_insights = sum(len(generate_mission_insights(data))
                           for data in precomputed['publications'].values())
        st.metric("AI-Generated Insights", f"{total_insights}+")

    with col4:
        tech_rich = sum(1 for data in precomputed['publications'].values()
                       if len(data.get('technical_terms', [])) > 8)
        st.metric("Technically Rich Studies", tech_rich)

    # Theme Maturity Progress
    st.subheader("Research Theme Maturity")

    theme_display_names = {
        'microgravity': 'Microgravity Effects',
        'radiation': 'Radiation Protection',
        'plant_biology': 'Food Production',
        'human_physiology': 'Crew Health',
        'microbiology': 'Microbiome Science',
        'life_support': 'Life Support Systems',
        'behavioral': 'Behavioral Health',
        'technology': 'Space Technology'
    }

    for theme_id, maturity_data in themes_maturity.items():
        display_name = theme_display_names.get(theme_id, theme_id.title())
        col1, col2, col3 = st.columns([2, 1, 3])

        with col1:
            st.write(f"**{display_name}**")
        with col2:
            st.write(f"{maturity_data['publications']} studies")
        with col3:
            progress = maturity_data['score'] / 100
            st.progress(progress)
            st.caption(f"Maturity: {maturity_data['score']}%")

    # Research Gaps Analysis
    st.subheader("Research Gaps Identification")

    # Identify less mature themes
    less_mature = [(theme_id, data) for theme_id, data in themes_maturity.items()
                  if data['score'] < 50]

    if less_mature:
        st.warning("Areas Needing More Research:")
        for theme_id, data in less_mature:
            display_name = theme_display_names.get(theme_id, theme_id.title())
            st.write(f"• {display_name} - Only {data['publications']} studies ({(100-data['score'])}% gap)")
    else:
        st.success("All major research areas have substantial coverage!")

def show_about():
    """Competition documentation"""
    st.header("NASA Space Apps Challenge 2025")

    st.markdown("""
    ## Bioscience Intelligence Platform

    **Transforming NASA Space Biology publications into actionable mission insights**

    ### The Challenge
    Exploring the Moon and Mars requires understanding how living systems respond to space.
    NASA's bioscience publications contain crucial knowledge, but it's challenging to extract
    actionable insights from this complex research.

    ### Our Solution
    An AI-powered platform that automatically analyzes, summarizes, and connects research findings
    to support scientists, mission architects, and program managers.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Innovations")
        st.markdown("""
        - AI-Powered Summarization: BART model generates concise summaries
        - Theme-Based Organization: 8 research themes with knowledge graphs
        - Mission Planning Insights: Automatic translation to actionable recommendations
        - Interactive Concept Mapping: Visualize technical relationships
        - Audio Synthesis: Make research accessible through audio
        - Multi-Perspective Views: Unified interface for all users
        """)

    with col2:
        st.subheader("Technical Stack")
        st.markdown("""
        - Frontend: Streamlit
        - AI/ML: Hugging Face Transformers (BART, Sentence-BERT)
        - Data Processing: Pandas, NumPy
        - Visualization: Plotly, NetworkX
        - NLP: spaCy, NLTK
        """)

    st.subheader("Data Sources")
    st.markdown("""
    - 608 NASA Space Biology Publications (PMC Open Access)
    - NASA Open Science Data Repository (OSDR)
    """)

def main():
    st.title("🚀 NASA Space Biology Intelligence Platform")
    st.markdown("**AI-powered exploration of NASA bioscience research**")

    if not precomputed.get('publications'):
        st.error("No precomputed data found. Please ensure 'nasa_precomputed_data' folder is in the same directory.")
        return


    # Initialize session state
    if 'selected_publication' not in st.session_state:
        st.session_state.selected_publication = None
    if 'current_section' not in st.session_state:
        st.session_state.current_section = "Dashboard Overview"
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []

    # Enhanced sidebar navigation
    with st.sidebar:
        nasa_logo = "nasa-logo.svg"  # or whatever your file is named
        st.image(nasa_logo, width=200)  # Adjust width as needed
        st.markdown("### NASA Bioscience Explorer")
        st.markdown("---")

        # Navigation
        nav_options = [
            "Dashboard Overview",
            "Impact Dashboard",
            "Research Themes",
            "Search & Explore",
            "Publication Analysis",
            "OSDR Data Integration",
            "Demo Scenarios",
            "About"
        ]

        for section in nav_options:
            if st.button(section, use_container_width=True, key=f"nav_{section}"):
                st.session_state.current_section = section
                st.rerun()

        st.markdown("---")


        # Current selection
        if st.session_state.selected_publication:
            st.markdown(f"**Selected Study:**")
            st.code(st.session_state.selected_publication)
            if st.button("Clear Selection", use_container_width=True):
                st.session_state.selected_publication = None
                st.rerun()

 
    # Section routing
    if st.session_state.current_section == "Dashboard Overview":
        show_dashboard_overview()
    elif st.session_state.current_section == "Impact Dashboard":
        show_impact_dashboard()
    elif st.session_state.current_section == "Research Themes":
        show_research_themes()
    elif st.session_state.current_section == "Search & Explore":
        show_search_explore()
    elif st.session_state.current_section == "Publication Analysis":
        show_publication_analysis()
    elif st.session_state.current_section == "OSDR Data Integration":  # ← ADD THIS
        show_osdr_integration()
    elif st.session_state.current_section == "Demo Scenarios":
        show_demo_scenarios()
    elif st.session_state.current_section == "About":
        show_about()

def show_dashboard_overview():
    """Dashboard overview"""
    st.header("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Publications", precomputed['metadata']['total_publications'])
    with col2:
        st.metric("Research Themes", len(precomputed['global_themes_network']['nodes']))
    with col3:
        # Count publications with mission-relevant insights
        mission_relevant = sum(1 for data in precomputed['publications'].values()
                             if any(theme in data.get('theme_analysis', {}).get('theme_counts', {})
                                  for theme in ['radiation', 'plant_biology', 'human_physiology', 'life_support']))
        st.metric("Mission-Critical Studies", mission_relevant)
    with col4:
        total_insights = sum(len(generate_mission_insights(data))
                           for data in precomputed['publications'].values())
        st.metric("AI-Generated Insights", f"{total_insights}+")

    # Quick actions for all users
    st.subheader("Quick Actions")

    actions = [
        ("Explore Research Themes", "themes"),
        ("Search Publications", "search"),
        ("View Impact Dashboard", "impact"),
        ("Try Demo Scenarios", "demo")
    ]

    cols = st.columns(len(actions))

    for idx, (action_name, action_key) in enumerate(actions):
        with cols[idx]:
            if st.button(action_name, use_container_width=True):
                if action_key == "impact":
                    st.session_state.current_section = "Impact Dashboard"
                elif action_key == "demo":
                    st.session_state.current_section = "Demo Scenarios"
                elif action_key == "themes":
                    st.session_state.current_section = "Research Themes"
                elif action_key == "search":
                    st.session_state.current_section = "Search & Explore"
                st.rerun()

    # Theme distribution
    st.subheader("Research Theme Distribution")
    theme_counts = {}
    for pub_data in precomputed['publications'].values():
        for theme, count in pub_data.get('theme_analysis', {}).get('theme_counts', {}).items():
            if count > 0:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1

    if theme_counts:
        theme_names = {
            'microgravity': 'Microgravity',
            'radiation': 'Radiation',
            'plant_biology': 'Plant Biology',
            'human_physiology': 'Human Physiology',
            'microbiology': 'Microbiology',
            'life_support': 'Life Support',
            'behavioral': 'Behavioral Health',
            'technology': 'Technology'
        }

        readable_themes = {theme_names.get(k, k): v for k, v in theme_counts.items()}

        fig = px.pie(
            values=list(readable_themes.values()),
            names=list(readable_themes.keys()),
            title="Publications Distribution Across Research Themes"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recent publications preview
    st.subheader("Recent Research Highlights")
    recent_pubs = list(precomputed['publications'].keys())[:6]

    for i, pmc_id in enumerate(recent_pubs):
        pub_data = precomputed['publications'][pmc_id]
        with st.expander(f"Publication {pmc_id}"):
            st.write("Abstract Summary:", pub_data.get('abstract_summary', 'No summary available'))

            # Show mission insights
            insights = generate_mission_insights(pub_data)
            if insights:
                st.write("Mission Relevance:")
                for insight in insights[:2]:  # Show top 2 insights
                    st.success(insight)

            tech_terms = pub_data.get('technical_terms', [])[:6]
            if tech_terms:
                st.write("Key Terms:", ", ".join(tech_terms))

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Analyze Publication", key=generate_unique_key("overview_analyze", pmc_id, i), use_container_width=True):
                    st.session_state.selected_publication = pmc_id
                    st.session_state.current_section = "Publication Analysis"
                    st.rerun()
            with col2:
                if st.button("Copy ID", key=generate_unique_key("overview_copy", pmc_id, i), use_container_width=True):
                    st.code(pmc_id)

def show_research_themes():
    """Global themes network"""
    st.header("Research Themes Network")

    global_net = precomputed['global_themes_network']

    theme_display_names = {
        'microgravity': 'Microgravity',
        'radiation': 'Radiation',
        'plant_biology': 'Plant Biology',
        'human_physiology': 'Human Physiology',
        'microbiology': 'Microbiology',
        'life_support': 'Life Support',
        'behavioral': 'Behavioral Health',
        'technology': 'Technology'
    }

    # Create network graph
    G = nx.Graph()
    for node in global_net['nodes']:
        display_name = theme_display_names.get(node['id'], node['id'].title())
        G.add_node(display_name, frequency=node['frequency'], original_id=node['id'])

    for edge in global_net['edges']:
        source_name = theme_display_names.get(edge['source'], edge['source'].title())
        target_name = theme_display_names.get(edge['target'], edge['target'].title())
        G.add_edge(source_name, target_name, weight=edge['weight'])

    # Create visualization
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, k=5, iterations=200, seed=42)

        # Edges
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(128,128,128,0.6)'),
            hoverinfo='none',
            mode='lines'
        )

        # Nodes
        node_x, node_y, node_text, node_size = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            freq = G.nodes[node]['frequency']
            node_text.append(f"{node}<br>{freq} publications")
            node_size.append(30 + (freq * 0.8))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],
            textposition="middle center",
            textfont=dict(size=14, color='black', family='Arial'),
            marker=dict(
                size=node_size,
                color=[G.nodes[node]['frequency'] for node in G.nodes()],
                colorscale='Viridis',
                line=dict(width=3, color='darkblue')
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Global Research Themes Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           height=700,
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))

        st.plotly_chart(fig, use_container_width=True)

    # Theme details
    st.subheader("Explore Publications by Theme")

    theme_options = [theme_display_names.get(node['id'], node['id'].title())
                    for node in global_net['nodes']]
    selected_display_theme = st.selectbox("Select a theme to explore:", theme_options)

    if selected_display_theme:
        original_theme = None
        for node in global_net['nodes']:
            if theme_display_names.get(node['id'], node['id'].title()) == selected_display_theme:
                original_theme = node['id']
                break

        if original_theme:
            publications = global_net['theme_publications'].get(original_theme, [])
            st.write(f"**{len(publications)} publications in {selected_display_theme}**")

            for i, pmc_id in enumerate(publications[:12]):
                pub_data = precomputed['publications'][pmc_id]
                with st.expander(f"Publication {pmc_id}"):
                    st.write("Abstract:", pub_data.get('abstract_summary', 'No summary available'))

                    # Show mission insights
                    insights = generate_mission_insights(pub_data)
                    if insights:
                        st.write("Mission Impact:")
                        for insight in insights:
                            if selected_display_theme.split()[0] in insight:
                                st.success(insight)

                    theme_terms = pub_data.get('theme_analysis', {}).get('theme_terms', {}).get(original_theme, [])
                    if theme_terms:
                        st.write("Theme Terms:", ", ".join(theme_terms[:8]))

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Analyze Publication", key=generate_unique_key("theme_analyze", pmc_id, i), use_container_width=True):
                            st.session_state.selected_publication = pmc_id
                            st.session_state.current_section = "Publication Analysis"
                            st.rerun()
                    with col2:
                        if st.button("Copy ID", key=generate_unique_key("theme_copy", pmc_id, i), use_container_width=True):
                            st.code(pmc_id)

def show_search_explore():
    """Search functionality"""
    st.header("Search & Explore Research")

    search_type = st.radio("Search by:", ["Theme", "Technical Term", "Keyword"], horizontal=True)

    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""

    results = []

    if search_type == "Theme":
        theme_options = list(precomputed['search_index']['by_theme'].keys())
        theme_display = {
            'microgravity': 'Microgravity Effects',
            'radiation': 'Radiation Protection',
            'plant_biology': 'Plant Biology',
            'human_physiology': 'Human Physiology',
            'microbiology': 'Microbiology',
            'life_support': 'Life Support Systems',
            'behavioral': 'Behavioral Health',
            'technology': 'Space Technology'
        }

        display_options = [theme_display.get(theme, theme.title()) for theme in theme_options]
        selected_display = st.selectbox("Select Research Theme:", display_options)

        if selected_display:
            for orig_theme, display_theme in theme_display.items():
                if display_theme == selected_display:
                    results = precomputed['search_index']['by_theme'].get(orig_theme, [])
                    st.session_state.search_results = results
                    st.session_state.search_query = selected_display
                    break

    elif search_type == "Technical Term":
        search_query = st.text_input("Enter technical term:")
        if search_query and search_query != st.session_state.search_query:
            results = precomputed['search_index']['by_technical_term'].get(search_query.lower(), [])
            st.session_state.search_results = results
            st.session_state.search_query = search_query
            if not results:
                st.info("No publications found with this exact term. Try similar terms.")

    else:  # Keyword
        search_query = st.text_input("Enter keyword or phrase:")
        if search_query and search_query != st.session_state.search_query:
            query_words = search_query.lower().split()
            all_results = set()

            for word in query_words:
                if word in precomputed['search_index']['by_text']:
                    all_results.update(precomputed['search_index']['by_text'][word])

            results = list(all_results)
            st.session_state.search_results = results
            st.session_state.search_query = search_query

    results = st.session_state.search_results

    # Display results
    if results:
        st.success(f"Found {len(results)} publications for '{st.session_state.search_query}'")

        unique_results = list(dict.fromkeys(results))
        sorted_results = sorted(
            unique_results,
            key=lambda x: len(precomputed['publications'][x].get('technical_terms', [])),
            reverse=True
        )

        for i, pmc_id in enumerate(sorted_results[:15]):
            pub_data = precomputed['publications'][pmc_id]

            with st.expander(f"Publication {pmc_id}"):
                st.write("Summary:", pub_data.get('abstract_summary', 'No summary available'))

                # Show mission insights for relevant results
                insights = generate_mission_insights(pub_data)
                if insights:
                    st.write("Mission Planning Insights:")
                    for insight in insights[:2]:
                        st.info(insight)

                tech_terms = pub_data.get('technical_terms', [])[:6]
                if tech_terms:
                    st.write("Key Terms:", ", ".join(tech_terms))

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Analyze Publication", key=generate_unique_key("search_analyze", pmc_id, i), use_container_width=True):
                        st.session_state.selected_publication = pmc_id
                        st.session_state.current_section = "Publication Analysis"
                        st.rerun()
                with col2:
                    if st.button("Copy ID", key=generate_unique_key("search_copy", pmc_id, i), use_container_width=True):
                        st.code(pmc_id)
    elif st.session_state.search_query:
        st.info("No results found. Try different search terms or browse by theme.")

def show_publication_analysis():
    """Individual publication analysis"""
    st.header("Publication Analysis")

    # Publication selector
    pub_options = list(precomputed['publications'].keys())

    if st.session_state.selected_publication and st.session_state.selected_publication in pub_options:
        default_index = pub_options.index(st.session_state.selected_publication)
    else:
        default_index = 0

    pmc_id = st.selectbox(
        "Select Publication:",
        pub_options,
        index=default_index
    )

    if pmc_id:
        st.info(f"Currently analyzing: {pmc_id}")
        pub_data = precomputed['publications'][pmc_id]

        # Original article link
        st.markdown(f"[View Original Article on PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmc_id.replace('PMC', '')}/)", unsafe_allow_html=True)

        # Quick actions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Technical Terms", len(pub_data.get('technical_terms', [])))
        with col2:
            active_themes = sum(1 for count in pub_data.get('theme_analysis', {}).get('theme_counts', {}).values() if count > 0)
            st.metric("Research Themes", active_themes)
        with col3:
            kg_nodes = len(pub_data.get('knowledge_graph', {}).get('nodes', []))
            st.metric("Concept Nodes", kg_nodes)

        # Mission Planning Insights
        st.subheader("Mission Planning Insights")
        insights = generate_mission_insights(pub_data)
        for insight in insights:
            st.success(insight)

        # Research Summary
        st.subheader("Research Summary")
        sections = [
            ('abstract', 'Abstract'),
            ('introduction', 'Introduction'),
            ('results', 'Results'),
            ('conclusion', 'Conclusion')
        ]

        for section_key, section_name in sections:
            summary = pub_data.get(f'{section_key}_summary')
            if summary and len(summary) > 20:
                with st.expander(section_name, expanded=(section_key == 'abstract')):
                    st.write(summary)

                    # Audio feature - REQUIRES BACKEND
                    if st.button(f"Listen to {section_name}", key=generate_unique_key("audio", pmc_id, section_key)):
                        with st.spinner("Generating audio..."):
                            audio_data = generate_audio(summary)  # Uses the new function we added
                            if audio_data:
                                audio_html = f'''
                                <audio controls style="width: 100%">
                                    <source src="data:audio/mp3;base64,{audio_data}" type="audio/mp3">
                                    Your browser does not support the audio element.
                                </audio>
                                '''
                                st.markdown(audio_html, unsafe_allow_html=True)
                                st.success("Audio ready! Click play to listen.")
                            else:
                                st.error("Audio generation failed. Please ensure the backend API is running.")

        # Knowledge Graph
        st.subheader("Technical Concepts Network")
        kg_data = pub_data.get('knowledge_graph', {})
        if kg_data and kg_data.get('nodes'):
            display_individual_knowledge_graph(kg_data)
        else:
            st.info("No significant technical concepts found for visualization")

        # Research Themes
        st.subheader("Research Themes")
        theme_counts = pub_data.get('theme_analysis', {}).get('theme_counts', {})
        if theme_counts:
            active_themes = [(theme, count) for theme, count in theme_counts.items() if count > 0]
            if active_themes:
                for theme, count in sorted(active_themes, key=lambda x: x[1], reverse=True):
                    theme_name = {
                        'microgravity': 'Microgravity',
                        'radiation': 'Radiation',
                        'plant_biology': 'Plant Biology',
                        'human_physiology': 'Human Physiology',
                        'microbiology': 'Microbiology',
                        'life_support': 'Life Support',
                        'behavioral': 'Behavioral Health',
                        'technology': 'Technology'
                    }.get(theme, theme.title())

                    st.write(f"**{theme_name}**: {count} related terms")
        
        # Technical Terms
        st.subheader("Key Technical Terms")
        tech_terms = pub_data.get('technical_terms', [])
        if tech_terms:
            cols = st.columns(3)
            terms_per_col = (len(tech_terms) + 2) // 3

            for i, col in enumerate(cols):
                start_idx = i * terms_per_col
                end_idx = min(start_idx + terms_per_col, len(tech_terms))
                with col:
                    for term in tech_terms[start_idx:end_idx]:
                        st.write(f"• {term}")
        else:
            st.info("No technical terms extracted")



def display_individual_knowledge_graph(kg_data):
    """Display knowledge graph for individual publication"""
    G = nx.Graph()

    for node in kg_data['nodes']:
        G.add_node(node['id'], frequency=node['frequency'])

    for edge in kg_data['edges']:
        G.add_edge(edge['source'], edge['target'])

    if len(G.nodes()) > 1:
        pos = nx.spring_layout(G, k=3, iterations=200, seed=42)

        pos.update({node: [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)] for node in G.nodes() if G.degree(node) == 0})
        edge_trace, node_trace = create_knowledge_graph_traces(G, pos)

        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Technical Concepts Relationship Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           height=650,
                           autosize=True,  
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, 2]),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, 2])
                       ))

        st.plotly_chart(fig, use_container_width=True)

        st.write("Concept Frequencies:")
        for node in sorted(kg_data['nodes'], key=lambda x: x['frequency'], reverse=True):
            st.write(f"• **{node['id']}**: appears {node['frequency']} times")
    else:
        st.info("Insufficient concepts for network visualization")

def create_knowledge_graph_traces(G, pos):
    """Create traces for knowledge graph visualization"""
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='rgba(100,100,100,0.6)'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text, node_size = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        freq = G.nodes[node]['frequency']
        node_text.append(f"{node}<br>Frequency: {freq}")
        node_size.append(25 + (freq * 2))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node for node in G.nodes()],
        textposition="middle center",
        textfont=dict(size=12, color='white'),
        marker=dict(
            size=node_size,
            color='lightcoral',
            line=dict(width=2, color='darkred')
        )
    )

    return edge_trace, node_trace


def show_osdr_integration():
    """OSDR Data Integration - Clean Metadata Viewer"""
    st.header("NASA OSDR Data Integration")
    st.markdown("Live connection to NASA Open Science Data Repository")

    # OSDR API configuration
    OSDR_API = "https://visualization.osdr.nasa.gov/biodata/api/v2"

    # Test connection
    try:
        test_response = requests.get(f"{OSDR_API}/datasets/", timeout=5)
        if test_response.status_code == 200:
            st.success("Connected to NASA OSDR API")
        else:
            st.warning("OSDR API accessible but datasets endpoint returned error")
    except Exception as e:
        st.error(f"Cannot connect to OSDR API: {e}")
        return

    # OSDR Metadata Viewer Section
    st.subheader("OSDR Metadata Explorer")

    # Fetch available datasets
    try:
        ds_resp = requests.get(f"{OSDR_API}/datasets/", timeout=10)
        if ds_resp.status_code == 200:
            datasets = list(ds_resp.json().keys())
            if datasets:
                selected_ds = st.selectbox("Select a dataset", datasets)

                # Fetch metadata for selected dataset
                if st.button("Fetch Metadata"):
                    try:
                        meta_resp = requests.get(f"{OSDR_API}/dataset/{selected_ds}/", timeout=10)
                        if meta_resp.status_code == 200:
                            ds_data = meta_resp.json().get(selected_ds, {})
                            st.success(f"Metadata retrieved for {selected_ds}")

                            # Render each top-level section in its own expander
                            for section, content in ds_data.items():
                                with st.expander(format_key(section), expanded=True):
                                    render_nested(content)
                        else:
                            st.error(f"Metadata fetch failed (status {meta_resp.status_code})")
                    except Exception as e:
                        st.error(f"Error fetching metadata: {e}")
            else:
                st.info("No datasets found in OSDR API response")
        else:
            st.error(f"Failed to fetch datasets (status {ds_resp.status_code})")
    except Exception as e:
        st.error(f"Error fetching datasets: {e}")

def format_key(key: str) -> str:
    """Format keys for display"""
    key = key.replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key).strip().title()
    return key

def render_nested(d, level=0):
    """Render dict/list recursively without nested expanders."""
    indent = " " * 4 * level
    if isinstance(d, dict):
        for k, v in d.items():
            key_fmt = format_key(k)
            if isinstance(v, dict) or isinstance(v, list):
                st.markdown(f"{indent}**{key_fmt}:**")
                render_nested(v, level + 1)
            else:
                st.markdown(f"{indent}**{key_fmt}:** {v}")
    elif isinstance(d, list):
        for i, item in enumerate(d, 1):
            if isinstance(item, dict) or isinstance(item, list):
                st.markdown(f"{indent}- Item {i}:")
                render_nested(item, level + 1)
            else:
                st.markdown(f"{indent}- {item}")
    else:
        st.markdown(f"{indent}{d}")

if __name__ == "__main__":
    main()
