"""
Preprocessing service for cleaning and lemmatizing Instagram bios.
Uses Natasha for Russian NLP processing.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
from rich.console import Console
from rich.progress import track

console = Console()

# Initialize Natasha components
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)


@dataclass
class ProcessedBio:
    original: str
    cleaned: str
    lemmatized: str
    tokens: List[str]
    entities: Dict[str, List[str]]  # professions, cities, etc.


@dataclass
class ProcessedProfile:
    """Combined processed data for a profile (bio + posts)"""
    username: str
    bio_clean: str
    bio_lemmatized: str
    captions_clean: List[str]
    all_hashtags: List[str]
    combined_text: str  # For clustering
    entities: Dict[str, List[str]]


class PreprocessingService:
    """Service for preprocessing Instagram bios"""

    def __init__(self):
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002500-\U00002BEF"  # misc symbols
            "\U00002702-\U000027B0"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u200d"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+",
            flags=re.UNICODE
        )

        # Common profession keywords (for entity extraction)
        self.profession_keywords = {
            'психолог', 'коуч', 'тренер', 'маркетолог', 'дизайнер',
            'разработчик', 'менеджер', 'предприниматель', 'блогер',
            'фотограф', 'стилист', 'визажист', 'врач', 'юрист',
            'бухгалтер', 'учитель', 'преподаватель', 'консультант',
            'специалист', 'эксперт', 'мастер', 'художник', 'писатель',
            'журналист', 'копирайтер', 'smm', 'таргетолог', 'продюсер'
        }

    def clean_text(self, text: str) -> str:
        """Remove emojis, links, and normalize text"""
        if not text:
            return ""

        # Remove emojis
        text = self.emoji_pattern.sub('', text)

        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)

        # Remove mentions and hashtags
        text = re.sub(r'[@#]\w+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Lowercase
        text = text.lower().strip()

        return text

    def lemmatize(self, text: str) -> Tuple[str, List[str]]:
        """Lemmatize Russian text using Natasha"""
        if not text:
            return "", []

        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)

        for token in doc.tokens:
            token.lemmatize(morph_vocab)

        lemmas = [token.lemma for token in doc.tokens if token.lemma]
        lemmatized_text = ' '.join(lemmas)

        return lemmatized_text, lemmas

    def extract_entities(self, text: str, lemmas: List[str]) -> Dict[str, List[str]]:
        """Extract named entities and keywords from bio"""
        entities = {
            'professions': [],
            'keywords': []
        }

        # Find professions
        text_lower = text.lower()
        for lemma in lemmas:
            if lemma in self.profession_keywords:
                entities['professions'].append(lemma)

        # Extract notable keywords (nouns only)
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)

        for token in doc.tokens:
            if hasattr(token, 'pos') and token.pos == 'NOUN':
                if len(token.text) > 3:
                    entities['keywords'].append(token.text.lower())

        return entities

    def process_bio(self, bio: str) -> ProcessedBio:
        """Full preprocessing pipeline for a single bio"""
        cleaned = self.clean_text(bio or "")
        lemmatized, tokens = self.lemmatize(cleaned)
        entities = self.extract_entities(cleaned, tokens)

        return ProcessedBio(
            original=bio or "",
            cleaned=cleaned,
            lemmatized=lemmatized,
            tokens=tokens,
            entities=entities
        )

    def process_batch(self, bios: List[str], show_progress: bool = True) -> List[ProcessedBio]:
        """Process a batch of bios"""
        results = []

        if show_progress:
            iterator = track(bios, description="Preprocessing bios...")
        else:
            iterator = bios

        for bio in iterator:
            results.append(self.process_bio(bio))

        # Stats
        non_empty = sum(1 for r in results if r.cleaned)
        console.print(f"[green]Processed {len(results)} bios ({non_empty} non-empty)[/green]")

        return results

    def filter_empty(self, bios: List[str]) -> List[str]:
        """Filter out empty or very short bios"""
        return [b for b in bios if b and len(b.strip()) > 10]

    def process_profile_with_posts(
        self,
        username: str,
        bio: Optional[str],
        posts: List[Dict]  # List of {caption, hashtags}
    ) -> ProcessedProfile:
        """
        Process profile with bio and posts, creating combined_text for clustering.

        Args:
            username: Instagram username
            bio: Profile biography
            posts: List of post dicts with 'caption' and 'hashtags' keys

        Returns:
            ProcessedProfile with combined_text for clustering
        """
        # Process bio
        bio_clean = self.clean_text(bio or "")
        bio_lemmatized, bio_tokens = self.lemmatize(bio_clean)

        # Process post captions
        captions_clean = []
        all_hashtags = []

        for post in posts:
            caption = post.get('caption') or ""
            caption_clean = self.clean_text(caption)
            if caption_clean:
                captions_clean.append(caption_clean)

            hashtags = post.get('hashtags') or []
            all_hashtags.extend([h.lower().replace('#', '') for h in hashtags])

        # Remove duplicates from hashtags
        all_hashtags = list(set(all_hashtags))

        # Create combined text for clustering
        combined_parts = []
        if bio_lemmatized:
            combined_parts.append(bio_lemmatized)
        if captions_clean:
            # Lemmatize captions
            for caption in captions_clean[:3]:  # Limit to 3 captions for clustering
                _, lemmas = self.lemmatize(caption)
                combined_parts.append(' '.join(lemmas))
        if all_hashtags:
            combined_parts.append(' '.join(all_hashtags[:20]))  # Top 20 hashtags

        combined_text = ' '.join(combined_parts)

        # Extract entities from combined text
        entities = self.extract_entities(combined_text, bio_tokens)

        return ProcessedProfile(
            username=username,
            bio_clean=bio_clean,
            bio_lemmatized=bio_lemmatized,
            captions_clean=captions_clean,
            all_hashtags=all_hashtags,
            combined_text=combined_text,
            entities=entities
        )

    def process_profiles_batch(
        self,
        profiles: List[Dict],  # List of {username, bio, posts: [{caption, hashtags}]}
        show_progress: bool = True
    ) -> List[ProcessedProfile]:
        """Process batch of profiles with posts"""
        results = []

        if show_progress:
            iterator = track(profiles, description="Preprocessing profiles...")
        else:
            iterator = profiles

        for profile in iterator:
            result = self.process_profile_with_posts(
                username=profile.get('username', ''),
                bio=profile.get('bio'),
                posts=profile.get('posts', [])
            )
            results.append(result)

        # Stats
        with_bio = sum(1 for r in results if r.bio_clean)
        with_posts = sum(1 for r in results if r.captions_clean)
        console.print(f"[green]Processed {len(results)} profiles[/green]")
        console.print(f"[dim]  - With bio: {with_bio}[/dim]")
        console.print(f"[dim]  - With posts: {with_posts}[/dim]")

        return results


if __name__ == "__main__":
    # Test
    service = PreprocessingService()

    test_bios = [
        "Психолог | Помогаю найти себя 🌟 Консультации онлайн ➡️ link.in/bio",
        "Мама двоих 👧👦 | Москва | Обожаю путешествия ✈️",
        "Маркетолог | SMM | Продвижение бизнеса 📈",
        "",
        "CEO @company | Предприниматель | 10+ лет в бизнесе",
    ]

    results = service.process_batch(test_bios)
    for r in results:
        if r.cleaned:
            console.print(f"[blue]Original:[/blue] {r.original[:50]}...")
            console.print(f"[green]Clean:[/green] {r.cleaned}")
            console.print(f"[yellow]Lemmas:[/yellow] {r.lemmatized}")
            console.print(f"[cyan]Entities:[/cyan] {r.entities}")
            console.print("---")
