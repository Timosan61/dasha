#!/usr/bin/env python3
"""
Fetch only NEW profiles (not in DB yet) to save API quota.
"""
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import init_db, get_session, Profile, Post
from services.apify_service import ApifyService
from rich.console import Console

console = Console()

def main():
    init_db()

    # Get existing usernames from DB
    with get_session() as session:
        existing = set(p[0] for p in session.query(Profile.username).all())
        console.print(f"[blue]В БД: {len(existing)} профилей[/blue]")

    # Load followers from latest file
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    followers_files = sorted(data_dir.glob('followers_*.json'), reverse=True)

    if not followers_files:
        console.print("[red]Файл с подписчиками не найден![/red]")
        return

    with open(followers_files[0]) as f:
        followers = json.load(f)

    # Filter: public + not in DB
    new_usernames = [
        f['username'] for f in followers
        if not f.get('is_private', False) and f['username'] not in existing
    ]

    console.print(f"[green]Новых для парсинга: {len(new_usernames)}[/green]")

    if not new_usernames:
        console.print("[yellow]Все профили уже спаршены![/yellow]")
        return

    # Fetch only new profiles
    apify = ApifyService()
    profiles = apify.fetch_profiles(new_usernames)

    # Save to DB
    posts_count = 0
    with get_session() as session:
        for p in profiles:
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
                source_profile='dasha_samoylina'
            )
            session.add(profile_obj)
            session.flush()

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

    console.print(f"[green]Сохранено {len(profiles)} новых профилей + {posts_count} постов[/green]")

if __name__ == "__main__":
    main()
