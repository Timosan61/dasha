"""
Apify service for fetching Instagram followers and their profiles.

Uses two Apify Actors:
1. datadoping/instagram-followers-scraper - get list of followers (usernames)
2. apify/instagram-profile-scraper - get profile details (bio, etc.)
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from apify_client import ApifyClient
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()

console = Console()

# Constants
FOLLOWERS_ACTOR = "datadoping/instagram-followers-scraper"
PROFILE_ACTOR = "apify/instagram-profile-scraper"
POST_SCRAPER_ACTOR = "apify/instagram-post-scraper"
COMMENTS_SCRAPER_ACTOR = "apify/instagram-comment-scraper"
BATCH_SIZE = 100  # Process profiles in batches
MAX_POSTS_PER_PROFILE = 5  # Only analyze last 5 posts


@dataclass
class FollowerData:
    username: str
    full_name: Optional[str] = None
    is_private: bool = False
    is_verified: bool = False


@dataclass
class PostData:
    post_id: str
    shortcode: str
    caption: Optional[str] = None
    hashtags: List[str] = None
    post_type: str = "Image"
    likes_count: int = 0
    comments_count: int = 0
    posted_at: Optional[str] = None

    def __post_init__(self):
        if self.hashtags is None:
            self.hashtags = []


@dataclass
class ProfileData:
    username: str
    full_name: Optional[str] = None
    biography: Optional[str] = None
    followers_count: int = 0
    following_count: int = 0
    posts_count: int = 0
    external_url: Optional[str] = None
    is_private: bool = False
    is_verified: bool = False
    is_business: bool = False
    latest_posts: List[PostData] = None

    def __post_init__(self):
        if self.latest_posts is None:
            self.latest_posts = []


@dataclass
class ReelData:
    post_id: str
    shortcode: str
    url: str
    caption: Optional[str] = None
    likes_count: int = 0
    comments_count: int = 0
    views_count: int = 0
    post_type: str = "Video"
    posted_at: Optional[str] = None
    owner_username: str = ""


@dataclass
class CommentData:
    comment_id: str
    text: str
    owner_username: str
    owner_full_name: Optional[str] = None
    likes_count: int = 0
    replies_count: int = 0
    is_reply: bool = False
    parent_comment_id: Optional[str] = None
    posted_at: Optional[str] = None


class ApifyService:
    """Service for fetching Instagram data via Apify with API key rotation"""

    # Active API keys (4, 5, 6 have credits)
    ACTIVE_KEYS = [4, 5, 6]

    def __init__(self):
        # Load all available API tokens
        self.tokens = []
        for key_num in self.ACTIVE_KEYS:
            token = os.getenv(f'APIFY_API_TOKEN_{key_num}')
            if token:
                self.tokens.append((key_num, token))

        # Fallback to default token if no numbered keys found
        if not self.tokens:
            default_token = os.getenv('APIFY_API_TOKEN')
            if default_token:
                self.tokens.append((0, default_token))

        if not self.tokens:
            raise ValueError("No APIFY_API_TOKEN found in environment")

        self.current_token_idx = 0
        self._init_client()
        self.data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[dim]Loaded {len(self.tokens)} API keys: {[k[0] for k in self.tokens]}[/dim]")

    def _init_client(self):
        """Initialize Apify client with current token"""
        key_num, token = self.tokens[self.current_token_idx]
        self.client = ApifyClient(token)
        self.current_key_num = key_num

    def _rotate_token(self) -> bool:
        """Rotate to next available token. Returns False if no more tokens."""
        if self.current_token_idx + 1 < len(self.tokens):
            self.current_token_idx += 1
            self._init_client()
            console.print(f"[yellow]Switched to API key #{self.current_key_num}[/yellow]")
            return True
        return False

    def _call_actor_with_retry(self, actor_id: str, run_input: dict) -> dict:
        """Call Apify actor with automatic key rotation on failure"""
        while True:
            try:
                console.print(f"[dim]Using API key #{self.current_key_num}[/dim]")
                run = self.client.actor(actor_id).call(run_input=run_input)
                return run
            except Exception as e:
                error_msg = str(e).lower()
                # Check if it's a quota/credits error
                if 'usage limit' in error_msg or 'quota' in error_msg or 'credit' in error_msg:
                    console.print(f"[red]API key #{self.current_key_num} exhausted: {e}[/red]")
                    if self._rotate_token():
                        continue
                    else:
                        raise ValueError("All API keys exhausted!")
                else:
                    raise

    def fetch_followers(self, username: str, max_count: int = 10000) -> List[FollowerData]:
        """
        Fetch list of followers for a given Instagram username.

        Args:
            username: Instagram username (without @)
            max_count: Maximum number of followers to fetch

        Returns:
            List of FollowerData objects
        """
        console.print(f"[bold blue]Fetching followers of @{username}...[/bold blue]")

        run_input = {
            "usernames": [username],
            "max_count": max_count
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running Apify Actor...", total=None)

            run = self._call_actor_with_retry(FOLLOWERS_ACTOR, run_input)
            progress.update(task, description="Processing results...")

            followers = []
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                followers.append(FollowerData(
                    username=item.get("username", ""),
                    full_name=item.get("full_name"),
                    is_private=item.get("is_private", False),
                    is_verified=item.get("is_verified", False),
                ))

        console.print(f"[green]Fetched {len(followers)} followers[/green]")

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.data_dir / f"followers_{username}_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([f.__dict__ for f in followers], f, ensure_ascii=False, indent=2)
        console.print(f"[dim]Saved to {filepath}[/dim]")

        return followers

    def fetch_profiles(self, usernames: List[str]) -> List[ProfileData]:
        """
        Fetch profile details for a list of usernames.
        Processes in batches to avoid API limits.

        Args:
            usernames: List of Instagram usernames

        Returns:
            List of ProfileData objects
        """
        console.print(f"[bold blue]Fetching {len(usernames)} profiles...[/bold blue]")

        profiles = []
        total_batches = (len(usernames) + BATCH_SIZE - 1) // BATCH_SIZE

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Processing batches (0/{total_batches})...", total=total_batches)

            for i in range(0, len(usernames), BATCH_SIZE):
                batch = usernames[i:i + BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                progress.update(task, description=f"Processing batch {batch_num}/{total_batches}...")

                run_input = {
                    "usernames": batch
                }

                try:
                    run = self._call_actor_with_retry(PROFILE_ACTOR, run_input)

                    for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                        # Extract posts (limit to MAX_POSTS_PER_PROFILE)
                        posts = []
                        latest_posts = item.get("latestPosts", [])[:MAX_POSTS_PER_PROFILE]
                        for post in latest_posts:
                            posts.append(PostData(
                                post_id=str(post.get("id", "")),
                                shortcode=post.get("shortCode", ""),
                                caption=post.get("caption"),
                                hashtags=post.get("hashtags", []),
                                post_type=post.get("type", "Image"),
                                likes_count=post.get("likesCount", 0),
                                comments_count=post.get("commentsCount", 0),
                                posted_at=post.get("timestamp"),
                            ))

                        profiles.append(ProfileData(
                            username=item.get("username", ""),
                            full_name=item.get("fullName"),
                            biography=item.get("biography"),
                            followers_count=item.get("followersCount", 0),
                            following_count=item.get("followsCount", 0),
                            posts_count=item.get("postsCount", 0),
                            external_url=item.get("externalUrl"),
                            is_private=item.get("private", False),
                            is_verified=item.get("verified", False),
                            is_business=item.get("isBusinessAccount", False),
                            latest_posts=posts,
                        ))
                except Exception as e:
                    console.print(f"[red]Error in batch {batch_num}: {e}[/red]")

                progress.advance(task)

        console.print(f"[green]Fetched {len(profiles)} profiles[/green]")

        # Count posts
        total_posts = sum(len(p.latest_posts) for p in profiles)
        console.print(f"[green]Total posts collected: {total_posts}[/green]")

        # Save to file (serialize posts too)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.data_dir / f"profiles_{timestamp}.json"

        def serialize_profile(p: ProfileData) -> dict:
            data = p.__dict__.copy()
            data['latest_posts'] = [post.__dict__ for post in p.latest_posts]
            return data

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([serialize_profile(p) for p in profiles], f, ensure_ascii=False, indent=2)
        console.print(f"[dim]Saved to {filepath}[/dim]")

        return profiles

    def fetch_all(self, target_username: str, max_followers: int = 3000) -> List[ProfileData]:
        """
        Complete pipeline: fetch followers, then fetch their profiles.

        Args:
            target_username: Target Instagram account
            max_followers: Maximum followers to fetch

        Returns:
            List of ProfileData objects with bio
        """
        # Step 1: Get followers list
        followers = self.fetch_followers(target_username, max_followers)

        # Filter out private accounts (can't get their bio anyway)
        public_usernames = [f.username for f in followers if not f.is_private]
        console.print(f"[yellow]Filtering: {len(public_usernames)} public accounts[/yellow]")

        # Step 2: Get profile details
        profiles = self.fetch_profiles(public_usernames)

        return profiles

    def fetch_posts(self, username: str, max_posts: int = 20, only_reels: bool = True) -> List[ReelData]:
        """
        Fetch posts/reels from a user's profile.

        Args:
            username: Instagram username
            max_posts: Maximum number of posts to fetch
            only_reels: If True, filter only Video/Reel posts

        Returns:
            List of ReelData objects
        """
        console.print(f"[bold blue]Fetching posts from @{username}...[/bold blue]")

        run_input = {
            "username": [username],
            "resultsLimit": max_posts
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running Instagram Post Scraper...", total=None)

            run = self._call_actor_with_retry(POST_SCRAPER_ACTOR, run_input)
            progress.update(task, description="Processing results...")

            reels = []
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                post_type = item.get("type", "Image")

                # Filter only reels/videos if requested
                if only_reels and post_type not in ["Video", "Reel", "Sidecar"]:
                    continue

                shortcode = item.get("shortCode", "")
                reels.append(ReelData(
                    post_id=str(item.get("id", "")),
                    shortcode=shortcode,
                    url=item.get("url", f"https://www.instagram.com/p/{shortcode}/"),
                    caption=item.get("caption"),
                    likes_count=item.get("likesCount", 0),
                    comments_count=item.get("commentsCount", 0),
                    views_count=item.get("videoViewCount", 0) or item.get("playCount", 0),
                    post_type=post_type,
                    posted_at=item.get("timestamp"),
                    owner_username=item.get("ownerUsername", username),
                ))

        console.print(f"[green]Fetched {len(reels)} reels/videos[/green]")

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.data_dir / f"reels_{username}_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([r.__dict__ for r in reels], f, ensure_ascii=False, indent=2)
        console.print(f"[dim]Saved to {filepath}[/dim]")

        return reels

    def fetch_comments(self, post_urls: List[str], max_comments_per_post: int = 500) -> Dict[str, List[CommentData]]:
        """
        Fetch comments for a list of posts.

        Args:
            post_urls: List of Instagram post URLs
            max_comments_per_post: Maximum comments to fetch per post

        Returns:
            Dict mapping post URL to list of CommentData
        """
        console.print(f"[bold blue]Fetching comments for {len(post_urls)} posts...[/bold blue]")

        run_input = {
            "directUrls": post_urls,
            "resultsLimit": max_comments_per_post
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running Instagram Comment Scraper...", total=None)

            run = self._call_actor_with_retry(COMMENTS_SCRAPER_ACTOR, run_input)
            progress.update(task, description="Processing results...")

            comments_by_post = {}
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                post_url = item.get("postUrl", "")

                if post_url not in comments_by_post:
                    comments_by_post[post_url] = []

                comments_by_post[post_url].append(CommentData(
                    comment_id=str(item.get("id", "")),
                    text=item.get("text", ""),
                    owner_username=item.get("ownerUsername", ""),
                    owner_full_name=item.get("ownerFullName"),
                    likes_count=item.get("likesCount", 0),
                    replies_count=item.get("repliesCount", 0),
                    is_reply=item.get("isReply", False),
                    parent_comment_id=item.get("parentCommentId"),
                    posted_at=item.get("timestamp"),
                ))

        total_comments = sum(len(c) for c in comments_by_post.values())
        console.print(f"[green]Fetched {total_comments} comments from {len(comments_by_post)} posts[/green]")

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.data_dir / f"comments_{timestamp}.json"

        # Serialize for JSON
        serialized = {url: [c.__dict__ for c in comments] for url, comments in comments_by_post.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)
        console.print(f"[dim]Saved to {filepath}[/dim]")

        return comments_by_post

    def load_from_file(self, filepath: str) -> List[Dict]:
        """Load previously saved data from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


if __name__ == "__main__":
    # Test run
    service = ApifyService()
    target = os.getenv('INSTAGRAM_TARGET', 'dasha_samoylina')

    # Fetch 10 followers for testing
    profiles = service.fetch_all(target, max_followers=10)
    for p in profiles[:5]:
        console.print(f"@{p.username}: {p.biography[:50] if p.biography else 'No bio'}...")
