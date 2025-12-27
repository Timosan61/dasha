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

from database import init_db, get_session, Profile, Post, Cluster, SegmentAnalysis, Reel, Comment, CommentAnalysis
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
@click.option('--max-followers', '-m', default=10000, help='Maximum followers to fetch')
@click.option('--skip-profiles', is_flag=True, help='Skip fetching profile details (bio)')
@click.option('--skip-existing/--no-skip-existing', default=True, help='Skip profiles already in DB')
def fetch(target: str, max_followers: int, skip_profiles: bool, skip_existing: bool):
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
    console.print(f"[yellow]{len(public_usernames)} public accounts[/yellow]")

    # Filter out existing profiles if --skip-existing is enabled
    if skip_existing:
        with get_session() as session:
            existing_usernames = set(
                u[0] for u in session.query(Profile.username).all()
            )
        new_usernames = [u for u in public_usernames if u not in existing_usernames]
        console.print(f"[green]Already in DB: {len(existing_usernames)}[/green]")
        console.print(f"[cyan]New to fetch: {len(new_usernames)}[/cyan]")

        if not new_usernames:
            console.print("[yellow]No new profiles to fetch. Use --no-skip-existing to force reload.[/yellow]")
            return

        public_usernames = new_usernames

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
            console.print("\n[bold]Step 2: Clustering with KMeans[/bold]")
            clusterer = ClusteringService(use_kmeans=True, n_clusters=10)

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


@cli.command()
@click.option('--target', '-t', default=None, help='Target Instagram username')
@click.option('--max-reels', '-m', default=20, help='Maximum reels to fetch')
@click.option('--max-comments', '-c', default=500, help='Maximum comments per reel')
def comments(target: str, max_reels: int, max_comments: int):
    """Fetch comments from reels for pain point analysis"""
    target = target or os.getenv('INSTAGRAM_TARGET', 'dasha_samoylina')

    console.print(Panel(f"[bold]Fetching comments from @{target} reels[/bold]", style="magenta"))

    init_db()
    apify = ApifyService()

    # Step 1: Fetch reels
    console.print("\n[bold]Step 1: Fetching reels[/bold]")
    reels = apify.fetch_posts(target, max_posts=max_reels, only_reels=True)

    if not reels:
        console.print("[red]No reels found![/red]")
        return

    console.print(f"[green]Found {len(reels)} reels[/green]")

    # Save reels to DB
    with get_session() as session:
        for reel in reels:
            existing = session.query(Reel).filter_by(post_id=reel.post_id).first()
            if existing:
                existing.likes_count = reel.likes_count
                existing.comments_count = reel.comments_count
                existing.views_count = reel.views_count
            else:
                session.add(Reel(
                    post_id=reel.post_id,
                    shortcode=reel.shortcode,
                    url=reel.url,
                    caption=reel.caption,
                    likes_count=reel.likes_count,
                    comments_count=reel.comments_count,
                    views_count=reel.views_count,
                    post_type=reel.post_type,
                    posted_at=datetime.fromisoformat(reel.posted_at.replace('Z', '+00:00')) if reel.posted_at else None,
                    owner_username=reel.owner_username,
                ))
        session.commit()

    # Step 2: Fetch comments
    console.print("\n[bold]Step 2: Fetching comments[/bold]")
    post_urls = [reel.url for reel in reels]
    comments_data = apify.fetch_comments(post_urls, max_comments_per_post=max_comments)

    # Save comments to DB
    saved_count = 0
    with get_session() as session:
        for url, url_comments in comments_data.items():
            # Find reel by URL
            reel = session.query(Reel).filter(Reel.url.contains(url.split('/')[-2])).first()
            if not reel:
                # Try by shortcode from URL
                shortcode = url.rstrip('/').split('/')[-1]
                reel = session.query(Reel).filter_by(shortcode=shortcode).first()

            if not reel:
                console.print(f"[yellow]Reel not found for URL: {url}[/yellow]")
                continue

            for comment in url_comments:
                existing = session.query(Comment).filter_by(comment_id=comment.comment_id).first()
                if not existing:
                    session.add(Comment(
                        reel_id=reel.id,
                        comment_id=comment.comment_id,
                        text=comment.text,
                        owner_username=comment.owner_username,
                        owner_full_name=comment.owner_full_name,
                        likes_count=comment.likes_count,
                        replies_count=comment.replies_count,
                        is_reply=comment.is_reply,
                        parent_comment_id=comment.parent_comment_id,
                        posted_at=datetime.fromisoformat(comment.posted_at.replace('Z', '+00:00')) if comment.posted_at else None,
                    ))
                    saved_count += 1
        session.commit()

    console.print(f"[green]Saved {saved_count} comments to database[/green]")

    # Summary
    with get_session() as session:
        total_reels = session.query(Reel).count()
        total_comments = session.query(Comment).count()
        questions = session.query(Comment).filter(Comment.text.like('%?%')).count()

    console.print(Panel(f"""
[bold green]Комментарии загружены![/bold green]

Рилсов: {total_reels}
Комментариев: {total_comments}
С вопросами (?): {questions}

Следующий шаг: python main.py analyze-comments
    """, style="green"))


@cli.command('analyze-comments')
def analyze_comments():
    """Analyze comments to extract pain points via GPT"""
    console.print(Panel("[bold]Analyzing comments with GPT[/bold]", style="cyan"))

    init_db()

    with get_session() as session:
        comments = session.query(Comment).all()
        reels = session.query(Reel).all()

        if not comments:
            console.print("[red]No comments found. Run 'comments' first.[/red]")
            return

        console.print(f"[blue]Loaded {len(comments)} comments from {len(reels)} reels[/blue]")

        # Group comments by reel
        comments_by_reel = {}
        for comment in comments:
            reel = comment.reel
            if reel.shortcode not in comments_by_reel:
                comments_by_reel[reel.shortcode] = {
                    'caption': reel.caption,
                    'comments': []
                }
            comments_by_reel[reel.shortcode]['comments'].append(comment.text)

        # Build prompt for GPT
        from services import AnalysisService

        analyzer = AnalysisService()

        # Format comments for analysis
        comments_text = ""
        for shortcode, data in comments_by_reel.items():
            caption_preview = (data['caption'][:100] + "...") if data['caption'] and len(data['caption']) > 100 else (data['caption'] or "без подписи")
            comments_text += f"\n## Рилс: {caption_preview}\n"
            comments_text += f"Комментарии ({len(data['comments'])}):\n"
            for c in data['comments'][:30]:  # Limit to 30 per reel
                if len(c) > 10:  # Skip very short
                    comments_text += f"- {c[:300]}\n"

        prompt = f"""Проанализируй комментарии к рилсам блогера про коммуникации.

КОММЕНТАРИИ:
{comments_text}

ЗАДАЧА: Найди в комментариях реальные боли аудитории. Ищи:
1. Вопросы, которые задают
2. Жалобы и проблемы
3. Ситуации, которые описывают
4. Что их триггерит/раздражает

ВАЖНО: Анализируй ТОЛЬКО реальные комментарии. Не придумывай боли.

Ответ в формате JSON:
{{
    "found_pains": [
        {{"pain": "описание боли", "evidence": "цитата из комментария", "frequency": "высокая/средняя/низкая"}},
        ...
    ],
    "found_questions": ["вопрос 1", "вопрос 2", ...],
    "main_topics": ["тема 1", "тема 2", ...],
    "audience_insights": "общие наблюдения об аудитории",
    "content_ideas": ["идея 1", "идея 2", ...]
}}"""

        try:
            response = analyzer.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Ты эксперт по анализу аудитории. Отвечай только на русском."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Print results
            console.print("\n[bold cyan]═══ НАЙДЕННЫЕ БОЛИ ═══[/bold cyan]")
            for pain in result.get('found_pains', []):
                freq_color = {"высокая": "red", "средняя": "yellow", "низкая": "dim"}.get(pain['frequency'], "white")
                console.print(f"\n[bold]{pain['pain']}[/bold] [{freq_color}]{pain['frequency']}[/{freq_color}]")
                console.print(f"  → _{pain['evidence']}_")

            console.print("\n[bold cyan]═══ ВОПРОСЫ АУДИТОРИИ ═══[/bold cyan]")
            for q in result.get('found_questions', []):
                console.print(f"  ❓ {q}")

            console.print("\n[bold cyan]═══ ОСНОВНЫЕ ТЕМЫ ═══[/bold cyan]")
            for t in result.get('main_topics', []):
                console.print(f"  • {t}")

            console.print(f"\n[bold cyan]═══ ИНСАЙТЫ ═══[/bold cyan]")
            console.print(result.get('audience_insights', ''))

            console.print(f"\n[bold cyan]═══ ИДЕИ КОНТЕНТА ═══[/bold cyan]")
            for idea in result.get('content_ideas', []):
                console.print(f"  💡 {idea}")

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(PROCESSED_DIR / f"comments_analysis_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            console.print(f"\n[dim]Saved to {PROCESSED_DIR}/comments_analysis_{timestamp}.json[/dim]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == '__main__':
    cli()
