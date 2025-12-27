"""
Streamlit Dashboard for Instagram Audience Analysis

Run with: streamlit run frontend/app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from database import get_session, Profile, Cluster, SegmentAnalysis

# Page config
st.set_page_config(
    page_title="Dasha - Audience Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .segment-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load data from database"""
    with get_session() as session:
        profiles = session.query(Profile).all()
        clusters = session.query(Cluster).all()
        analyses = session.query(SegmentAnalysis).all()

        profiles_df = pd.DataFrame([{
            'username': p.username,
            'full_name': p.full_name,
            'bio': p.bio,
            'bio_clean': p.bio_clean,
            'followers_count': p.followers_count,
            'following_count': p.following_count,
            'posts_count': p.posts_count,
            'external_url': p.external_url,
            'cluster_id': p.cluster_id,
            'cluster_name': p.cluster.name if p.cluster else 'Не кластеризован',
        } for p in profiles])

        clusters_data = []
        for c in clusters:
            analysis = c.analysis
            clusters_data.append({
                'id': c.id,
                'topic_id': c.topic_id,
                'name': c.name,
                'keywords': c.keywords or [],
                'size': c.size,
                'size_percent': c.size_percent,
                'segment_name': analysis.segment_name if analysis else None,
                'main_pain': analysis.main_pain if analysis else None,
                'triggers': analysis.triggers if analysis else None,
                'desired_outcome': analysis.desired_outcome if analysis else None,
                'client_phrase': analysis.client_phrase if analysis else None,
            })

        clusters_df = pd.DataFrame(clusters_data)

    return profiles_df, clusters_df


def main():
    # Sidebar
    st.sidebar.title("📊 Dasha")
    st.sidebar.markdown("Анализ аудитории Instagram")

    page = st.sidebar.radio(
        "Навигация",
        ["🏠 Обзор", "📈 Кластеры", "👥 Профили", "📥 Экспорт"]
    )

    # Load data
    try:
        profiles_df, clusters_df = load_data()
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        st.info("Запустите `python main.py fetch` и `python main.py analyze` для сбора данных")
        return

    if profiles_df.empty:
        st.warning("Нет данных. Запустите `python main.py fetch` для сбора данных.")
        return

    # Pages
    if page == "🏠 Обзор":
        show_overview(profiles_df, clusters_df)
    elif page == "📈 Кластеры":
        show_clusters(profiles_df, clusters_df)
    elif page == "👥 Профили":
        show_profiles(profiles_df, clusters_df)
    elif page == "📥 Экспорт":
        show_export(profiles_df, clusters_df)


def show_overview(profiles_df: pd.DataFrame, clusters_df: pd.DataFrame):
    """Overview page with key metrics"""
    st.title("🏠 Обзор аудитории")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Всего профилей", len(profiles_df))

    with col2:
        bio_count = profiles_df['bio'].apply(lambda x: bool(x and str(x).strip())).sum()
        st.metric("С биографией", bio_count)

    with col3:
        st.metric("Кластеров", len(clusters_df))

    with col4:
        analyzed = clusters_df['segment_name'].notna().sum()
        st.metric("Проанализировано", analyzed)

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Распределение по кластерам")
        if not clusters_df.empty:
            fig = px.pie(
                clusters_df,
                values='size',
                names='name',
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Размер кластеров")
        if not clusters_df.empty:
            fig = px.bar(
                clusters_df.sort_values('size', ascending=True),
                x='size',
                y='name',
                orientation='h',
                color='size',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Top keywords
    st.subheader("Ключевые слова")
    all_keywords = []
    for kw_list in clusters_df['keywords'].dropna():
        if isinstance(kw_list, list):
            all_keywords.extend(kw_list[:5])

    if all_keywords:
        keyword_counts = pd.Series(all_keywords).value_counts().head(20)
        fig = px.bar(
            x=keyword_counts.values,
            y=keyword_counts.index,
            orientation='h',
            labels={'x': 'Частота', 'y': 'Ключевое слово'}
        )
        st.plotly_chart(fig, use_container_width=True)


def show_clusters(profiles_df: pd.DataFrame, clusters_df: pd.DataFrame):
    """Clusters page with detailed analysis"""
    st.title("📈 Анализ кластеров")

    if clusters_df.empty:
        st.warning("Нет данных о кластерах. Запустите анализ.")
        return

    # Cluster selector
    cluster_names = clusters_df['name'].tolist()
    selected_cluster = st.selectbox("Выберите кластер", cluster_names)

    cluster_row = clusters_df[clusters_df['name'] == selected_cluster].iloc[0]

    # Cluster info
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📊 {cluster_row['name']}")

        # Segment analysis
        if cluster_row['segment_name']:
            st.markdown(f"### {cluster_row['segment_name']}")

            st.markdown("**🎯 Основная боль:**")
            st.info(cluster_row['main_pain'])

            st.markdown("**⚡ Триггерные ситуации:**")
            if cluster_row['triggers']:
                for trigger in cluster_row['triggers']:
                    st.markdown(f"- {trigger}")

            st.markdown("**🎯 Желаемый результат:**")
            st.success(cluster_row['desired_outcome'])

            st.markdown("**💬 Типичная фраза клиента:**")
            st.warning(f"_{cluster_row['client_phrase']}_")
        else:
            st.warning("GPT-анализ не выполнен для этого кластера")

    with col2:
        st.metric("Профилей", cluster_row['size'])
        st.metric("Доля", f"{cluster_row['size_percent']}%")

        st.markdown("**Ключевые слова:**")
        if cluster_row['keywords']:
            for kw in cluster_row['keywords'][:10]:
                st.markdown(f"- {kw}")

    # Sample profiles
    st.markdown("---")
    st.subheader("Примеры профилей в кластере")

    cluster_profiles = profiles_df[profiles_df['cluster_id'] == cluster_row['id']].head(10)
    if not cluster_profiles.empty:
        for _, row in cluster_profiles.iterrows():
            with st.expander(f"@{row['username']} — {row['full_name'] or 'N/A'}"):
                st.markdown(f"**Bio:** {row['bio'] or 'Нет'}")
                st.markdown(f"**Подписчиков:** {row['followers_count']:,}")


def show_profiles(profiles_df: pd.DataFrame, clusters_df: pd.DataFrame):
    """Profiles page with search and filters"""
    st.title("👥 Профили подписчиков")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        search = st.text_input("🔍 Поиск по bio", "")

    with col2:
        cluster_filter = st.selectbox(
            "Кластер",
            ["Все"] + clusters_df['name'].tolist()
        )

    with col3:
        min_followers = st.number_input("Мин. подписчиков", min_value=0, value=0)

    # Apply filters
    filtered = profiles_df.copy()

    if search:
        filtered = filtered[
            filtered['bio'].str.contains(search, case=False, na=False) |
            filtered['username'].str.contains(search, case=False, na=False)
        ]

    if cluster_filter != "Все":
        filtered = filtered[filtered['cluster_name'] == cluster_filter]

    if min_followers > 0:
        filtered = filtered[filtered['followers_count'] >= min_followers]

    st.markdown(f"**Найдено:** {len(filtered)} профилей")

    # Display table
    st.dataframe(
        filtered[['username', 'full_name', 'bio', 'followers_count', 'cluster_name']].head(100),
        use_container_width=True,
        height=500
    )


def show_export(profiles_df: pd.DataFrame, clusters_df: pd.DataFrame):
    """Export page"""
    st.title("📥 Экспорт данных")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Профили")

        csv_profiles = profiles_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Скачать CSV",
            csv_profiles,
            "profiles.csv",
            "text/csv"
        )

        json_profiles = profiles_df.to_json(orient='records', force_ascii=False)
        st.download_button(
            "⬇️ Скачать JSON",
            json_profiles,
            "profiles.json",
            "application/json"
        )

    with col2:
        st.subheader("Сегменты")

        csv_clusters = clusters_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Скачать CSV",
            csv_clusters,
            "segments.csv",
            "text/csv"
        )

        json_clusters = clusters_df.to_json(orient='records', force_ascii=False)
        st.download_button(
            "⬇️ Скачать JSON",
            json_clusters,
            "segments.json",
            "application/json"
        )

    # Full report
    st.markdown("---")
    st.subheader("Полный отчёт")

    report_lines = ["# Анализ аудитории Instagram\n"]
    report_lines.append(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    report_lines.append(f"**Всего профилей:** {len(profiles_df)}\n")
    report_lines.append(f"**Кластеров:** {len(clusters_df)}\n\n")

    for _, row in clusters_df.iterrows():
        report_lines.append(f"## {row['segment_name'] or row['name']}\n")
        report_lines.append(f"**Размер:** {row['size']} ({row['size_percent']}%)\n")
        if row['main_pain']:
            report_lines.append(f"**Боль:** {row['main_pain']}\n")
        if row['client_phrase']:
            report_lines.append(f"**Фраза:** _{row['client_phrase']}_\n")
        report_lines.append("\n---\n")

    report = "\n".join(report_lines)

    st.download_button(
        "⬇️ Скачать Markdown отчёт",
        report,
        "report.md",
        "text/markdown"
    )


if __name__ == "__main__":
    main()
