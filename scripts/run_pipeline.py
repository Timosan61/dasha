#!/usr/bin/env python3
"""
Full pipeline script for Instagram audience analysis.

Usage:
    python scripts/run_pipeline.py [--max-followers 3000] [--target dasha_samoylina]
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

from database import init_db, get_session, Profile, Cluster, SegmentAnalysis
from services import ApifyService, PreprocessingService, ClusteringService, AnalysisService

load_dotenv()

console = Console()


@click.command()
@click.option('--target', '-t', default=None, help='Target Instagram username')
@click.option('--max-followers', '-m', default=3000, help='Maximum followers to fetch')
@click.option('--skip-fetch', is_flag=True, help='Skip fetching (use existing data)')
def run_pipeline(target: str, max_followers: int, skip_fetch: bool):
    """Run full analysis pipeline"""
    target = target or os.getenv('INSTAGRAM_TARGET', 'dasha_samoylina')

    console.print(Panel(
        f"[bold]Instagram Audience Analysis Pipeline[/bold]\n\n"
        f"Target: @{target}\n"
        f"Max followers: {max_followers}",
        style="blue"
    ))

    # Initialize database
    init_db()

    # Step 1: Fetch data
    if not skip_fetch:
        console.print("\n[bold cyan]═══ STEP 1: Fetching Data from Apify ═══[/bold cyan]")
        apify = ApifyService()
        profiles = apify.fetch_all(target, max_followers)

        # Save to database
        from datetime import datetime
        with get_session() as session:
            for p in profiles:
                existing = session.query(Profile).filter_by(username=p.username).first()
                if not existing:
                    session.add(Profile(
                        username=p.username,
                        full_name=p.full_name,
                        bio=p.biography,
                        followers_count=p.followers_count,
                        following_count=p.following_count,
                        posts_count=p.posts_count,
                        external_url=p.external_url,
                        is_private=p.is_private,
                        is_verified=p.is_verified,
                        is_business=p.is_business,
                        source_profile=target
                    ))
        console.print(f"[green]Saved {len(profiles)} profiles[/green]")
    else:
        console.print("[yellow]Skipping fetch (using existing data)[/yellow]")

    # Step 2: Preprocessing
    console.print("\n[bold cyan]═══ STEP 2: Preprocessing Bios ═══[/bold cyan]")
    preprocessor = PreprocessingService()

    with get_session() as session:
        profiles = session.query(Profile).filter(Profile.bio.isnot(None)).all()
        bios = [p.bio for p in profiles]
        processed = preprocessor.process_batch(bios)

        for profile, proc in zip(profiles, processed):
            profile.bio_clean = proc.lemmatized

    # Step 3: Clustering
    console.print("\n[bold cyan]═══ STEP 3: Clustering with BERTopic ═══[/bold cyan]")
    clusterer = ClusteringService()

    with get_session() as session:
        profiles = session.query(Profile).filter(Profile.bio_clean.isnot(None)).all()
        docs = [p.bio_clean for p in profiles]

        if len(docs) < 20:
            console.print("[red]Not enough profiles for clustering (need at least 20)[/red]")
            return

        topics, clusters = clusterer.fit(docs)

        # Save clusters
        for result in clusters:
            existing = session.query(Cluster).filter_by(topic_id=result.topic_id).first()
            if not existing:
                session.add(Cluster(
                    topic_id=result.topic_id,
                    name=result.name,
                    keywords=result.keywords,
                    size=result.size,
                    size_percent=result.size_percent
                ))

        # Assign topics
        session.commit()
        for profile, topic_id in zip(profiles, topics):
            if topic_id >= 0:
                cluster = session.query(Cluster).filter_by(topic_id=topic_id).first()
                if cluster:
                    profile.cluster_id = cluster.id

        clusterer.save_model()

    # Step 4: GPT Analysis
    console.print("\n[bold cyan]═══ STEP 4: GPT-4o-mini Pain Point Analysis ═══[/bold cyan]")
    analyzer = AnalysisService()

    with get_session() as session:
        clusters = session.query(Cluster).all()
        cluster_data = []

        for cluster in clusters:
            cluster_profiles = session.query(Profile).filter_by(cluster_id=cluster.id).all()
            bios = [p.bio for p in cluster_profiles if p.bio]
            if bios:
                cluster_data.append({
                    'id': cluster.id,
                    'bios': bios,
                    'keywords': cluster.keywords or []
                })

        results = analyzer.analyze_all_clusters(cluster_data)

        # Save analysis
        for result in results:
            existing = session.query(SegmentAnalysis).filter_by(cluster_id=result.cluster_id).first()
            if not existing:
                session.add(SegmentAnalysis(
                    cluster_id=result.cluster_id,
                    segment_name=result.segment_name,
                    main_pain=result.main_pain,
                    triggers=result.triggers,
                    desired_outcome=result.desired_outcome,
                    client_phrase=result.client_phrase,
                    raw_response=result.raw_response
                ))

        # Print report
        report = analyzer.format_report(results)
        console.print(report)

    console.print(Panel(
        "[bold green]Pipeline Complete![/bold green]\n\n"
        "Run `python main.py dashboard` to view results",
        style="green"
    ))


if __name__ == '__main__':
    run_pipeline()
