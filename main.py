#!/usr/bin/env python3
"""
Dasha - Instagram Audience Analyzer

CLI tool for analyzing Instagram followers using:
- Apify for data collection
- BERTopic for clustering
- GPT-4o-mini for pain point extraction

Usage:
    python main.py fetch          - Fetch followers from Instagram
    python main.py analyze        - Run full analysis pipeline
    python main.py dashboard      - Launch Streamlit dashboard
    python main.py export         - Export results to CSV/JSON
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from database import init_db, get_session, Profile, Post, Cluster, SegmentAnalysis
from services import ApifyService, PreprocessingService, ClusteringService, AnalysisService

load_dotenv()

console = Console()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'


@click.group()
def cli():
    """Dasha - Instagram Audience Analyzer"""
    pass


@cli.command()
@click.option('--target', '-t', default=None, help='Target Instagram username')
@click.option('--max-followers', '-m', default=3000, help='Maximum followers to fetch')
@click.option('--skip-profiles', is_flag=True, help='Skip fetching profile details (bio)')
def fetch(target: str, max_followers: int, skip_profiles: bool):
    """Fetch Instagram followers and their profiles via Apify"""
    target = target or os.getenv('INSTAGRAM_TARGET', 'dasha_samoylina')

    console.print(Panel(f"[bold]Fetching followers of @{target}[/bold]", style="blue"))

    init_db()
    apify = ApifyService()

    # Step 1: Fetch followers list
    followers = apify.fetch_followers(target, max_followers)
    console.print(f"[green]Got {len(followers)} followers[/green]")

    if skip_profiles:
        console.print("[yellow]Skipping profile details (--skip-profiles)[/yellow]")
        return

    # Step 2: Fetch profiles with bio
    public_usernames = [f.username for f in followers if not f.is_private]
    console.print(f"[yellow]{len(public_usernames)} public accounts to fetch[/yellow]")

    profiles = apify.fetch_profiles(public_usernames)

    # Step 3: Save to database (profiles + posts)
    posts_count = 0
    with get_session() as session:
        for p in profiles:
            existing = session.query(Profile).filter_by(username=p.username).first()
            if existing:
                # Update profile
                existing.bio = p.biography
                existing.followers_count = p.followers_count
                existing.following_count = p.following_count
                existing.posts_count = p.posts_count
                existing.external_url = p.external_url
                existing.fetched_at = datetime.utcnow()
                profile_obj = existing
                # Delete old posts to replace with fresh ones
                session.query(Post).filter_by(profile_id=existing.id).delete()
            else:
                # Insert new profile
                profile_obj = Profile(
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
                )
                session.add(profile_obj)
                session.flush()  # Get ID for the new profile

            # Add posts
            for post_data in p.latest_posts:
                post = Post(
                    profile_id=profile_obj.id,
                    post_id=post_data.post_id,
                    shortcode=post_data.shortcode,
                    caption=post_data.caption,
                    hashtags=post_data.hashtags,
                    post_type=post_data.post_type,
                    likes_count=post_data.likes_count,
                    comments_count=post_data.comments_count,
                    posted_at=datetime.fromisoformat(post_data.posted_at.replace('Z', '+00:00')) if post_data.posted_at else None
                )
                session.add(post)
                posts_count += 1

        session.commit()

    console.print(f"[green]Saved {len(profiles)} profiles + {posts_count} posts to database[/green]")


@cli.command()
@click.option('--skip-preprocessing', is_flag=True, help='Skip preprocessing step')
@click.option('--skip-clustering', is_flag=True, help='Skip clustering step')
@click.option('--skip-gpt', is_flag=True, help='Skip GPT analysis step')
def analyze(skip_preprocessing: bool, skip_clustering: bool, skip_gpt: bool):
    """Run full analysis pipeline: preprocess → cluster → GPT analysis"""
    console.print(Panel("[bold]Running Analysis Pipeline[/bold]", style="green"))

    init_db()

    with get_session() as session:
        # Load profiles
        profiles = session.query(Profile).filter(Profile.bio.isnot(None)).all()
        console.print(f"[blue]Loaded {len(profiles)} profiles with bio[/blue]")

        if not profiles:
            console.print("[red]No profiles found. Run 'fetch' first.[/red]")
            return

        # Step 1: Preprocessing (bio + posts → combined_text)
        if not skip_preprocessing:
            console.print("\n[bold]Step 1: Preprocessing (bio + posts)[/bold]")
            preprocessor = PreprocessingService()

            # Build profile data with posts
            profiles_data = []
            for profile in profiles:
                posts_list = []
                for post in profile.posts:
                    posts_list.append({
                        'caption': post.caption,
                        'hashtags': post.hashtags or []
                    })
                profiles_data.append({
                    'username': profile.username,
                    'bio': profile.bio,
                    'posts': posts_list
                })

            # Process with posts
            processed = preprocessor.process_profiles_batch(profiles_data)

            # Update profiles with combined_text
            for profile, proc in zip(profiles, processed):
                profile.bio_clean = proc.bio_lemmatized
                profile.combined_text = proc.combined_text

                # Update caption_clean for posts
                for post, cap_clean in zip(profile.posts, proc.captions_clean):
                    post.caption_clean = cap_clean

            session.commit()
            console.print("[green]Preprocessing complete[/green]")

        # Step 2: Clustering
        if not skip_clustering:
            console.print("\n[bold]Step 2: Clustering with BERTopic[/bold]")
            clusterer = ClusteringService()

            # Use combined_text (bio + posts + hashtags) for clustering
            docs = [p.combined_text or p.bio_clean or p.bio for p in profiles if p.bio]
            topics, clusters = clusterer.fit(docs)

            # Save clusters to DB
            for result in clusters:
                existing = session.query(Cluster).filter_by(topic_id=result.topic_id).first()
                if existing:
                    existing.name = result.name
                    existing.keywords = result.keywords
                    existing.size = result.size
                    existing.size_percent = result.size_percent
                else:
                    session.add(Cluster(
                        topic_id=result.topic_id,
                        name=result.name,
                        keywords=result.keywords,
                        size=result.size,
                        size_percent=result.size_percent
                    ))
            session.commit()

            # Assign topics to profiles
            valid_profiles = [p for p in profiles if p.bio]
            for profile, topic_id in zip(valid_profiles, topics):
                if topic_id >= 0:
                    cluster = session.query(Cluster).filter_by(topic_id=topic_id).first()
                    if cluster:
                        profile.cluster_id = cluster.id
            session.commit()

            # Save model
            clusterer.save_model()
            console.print("[green]Clustering complete[/green]")

        # Step 3: GPT Analysis
        if not skip_gpt:
            console.print("\n[bold]Step 3: GPT-4o-mini Analysis[/bold]")
            analyzer = AnalysisService()

            clusters = session.query(Cluster).all()
            cluster_data = []

            for cluster in clusters:
                cluster_profiles = session.query(Profile).filter_by(cluster_id=cluster.id).all()
                bios = [p.bio for p in cluster_profiles if p.bio]

                # Collect captions and hashtags from posts
                captions = []
                hashtags = []
                for profile in cluster_profiles:
                    for post in profile.posts:
                        if post.caption:
                            captions.append(post.caption)
                        if post.hashtags:
                            hashtags.extend(post.hashtags)

                if bios:
                    cluster_data.append({
                        'id': cluster.id,
                        'bios': bios,
                        'keywords': cluster.keywords or [],
                        'captions': captions,
                        'hashtags': hashtags
                    })

            results = analyzer.analyze_all_clusters(cluster_data)

            # Save to DB
            for result in results:
                existing = session.query(SegmentAnalysis).filter_by(cluster_id=result.cluster_id).first()
                if existing:
                    existing.segment_name = result.segment_name
                    existing.portrait = result.portrait
                    existing.main_pain = result.main_pain
                    existing.triggers = result.triggers
                    existing.desired_outcome = result.desired_outcome
                    existing.client_phrase = result.client_phrase
                    existing.content_interests = result.content_interests
                    existing.raw_response = result.raw_response
                else:
                    session.add(SegmentAnalysis(
                        cluster_id=result.cluster_id,
                        segment_name=result.segment_name,
                        portrait=result.portrait,
                        main_pain=result.main_pain,
                        triggers=result.triggers,
                        desired_outcome=result.desired_outcome,
                        client_phrase=result.client_phrase,
                        content_interests=result.content_interests,
                        raw_response=result.raw_response
                    ))
            session.commit()

            # Print report
            report = analyzer.format_report(results)
            console.print(report)

    console.print(Panel("[bold green]Analysis Complete![/bold green]"))


@cli.command()
@click.option('--port', '-p', default=8501, help='Streamlit port')
def dashboard(port: int):
    """Launch Streamlit dashboard"""
    import subprocess
    app_path = PROJECT_ROOT / 'frontend' / 'app.py'
    subprocess.run(['streamlit', 'run', str(app_path), '--server.port', str(port)])


@cli.command()
@click.option('--format', '-f', type=click.Choice(['csv', 'json', 'both']), default='both')
def export(format: str):
    """Export results to CSV/JSON files"""
    import pandas as pd

    console.print(Panel("[bold]Exporting Data[/bold]", style="cyan"))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with get_session() as session:
        # Export profiles with posts
        profiles = session.query(Profile).all()
        profiles_data = [{
            'username': p.username,
            'full_name': p.full_name,
            'bio': p.bio,
            'bio_clean': p.bio_clean,
            'combined_text': p.combined_text,
            'followers_count': p.followers_count,
            'following_count': p.following_count,
            'posts_count': p.posts_count,
            'external_url': p.external_url,
            'cluster_id': p.cluster_id,
            'cluster_name': p.cluster.name if p.cluster else None,
            'posts': [{
                'caption': post.caption,
                'hashtags': post.hashtags,
                'post_type': post.post_type,
                'likes_count': post.likes_count,
            } for post in p.posts]
        } for p in profiles]

        # Export clusters with analysis
        clusters = session.query(Cluster).all()
        clusters_data = []
        for c in clusters:
            analysis = c.analysis
            clusters_data.append({
                'topic_id': c.topic_id,
                'name': c.name,
                'keywords': c.keywords,
                'size': c.size,
                'size_percent': c.size_percent,
                'segment_name': analysis.segment_name if analysis else None,
                'portrait': analysis.portrait if analysis else None,
                'main_pain': analysis.main_pain if analysis else None,
                'triggers': analysis.triggers if analysis else None,
                'desired_outcome': analysis.desired_outcome if analysis else None,
                'client_phrase': analysis.client_phrase if analysis else None,
                'content_interests': analysis.content_interests if analysis else None,
            })

    # Save files
    if format in ['csv', 'both']:
        df_profiles = pd.DataFrame(profiles_data)
        df_profiles.to_csv(PROCESSED_DIR / f'profiles_{timestamp}.csv', index=False)
        console.print(f"[green]Saved profiles CSV[/green]")

        df_clusters = pd.DataFrame(clusters_data)
        df_clusters.to_csv(PROCESSED_DIR / f'clusters_{timestamp}.csv', index=False)
        console.print(f"[green]Saved clusters CSV[/green]")

    if format in ['json', 'both']:
        with open(PROCESSED_DIR / f'profiles_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
        console.print(f"[green]Saved profiles JSON[/green]")

        with open(PROCESSED_DIR / f'segments_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(clusters_data, f, ensure_ascii=False, indent=2)
        console.print(f"[green]Saved segments JSON[/green]")

    console.print(f"[dim]Files saved to {PROCESSED_DIR}[/dim]")


@cli.command()
def init():
    """Initialize database"""
    init_db()
    console.print("[green]Database initialized[/green]")


if __name__ == '__main__':
    cli()
