from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Profile(Base):
    """Instagram profile data"""
    __tablename__ = 'profiles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255))
    bio = Column(Text)
    bio_clean = Column(Text)  # Lemmatized/preprocessed bio
    followers_count = Column(Integer, default=0)
    following_count = Column(Integer, default=0)
    posts_count = Column(Integer, default=0)
    external_url = Column(String(512))
    is_private = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    is_business = Column(Boolean, default=False)

    # Combined text for clustering (bio + captions + hashtags)
    combined_text = Column(Text)

    # Clustering
    cluster_id = Column(Integer, ForeignKey('clusters.id'), nullable=True)
    cluster = relationship('Cluster', back_populates='profiles')

    # Posts relationship
    posts = relationship('Post', back_populates='profile', cascade='all, delete-orphan')

    # Metadata
    fetched_at = Column(DateTime, default=datetime.utcnow)
    source_profile = Column(String(255), default='dasha_samoylina')

    def __repr__(self):
        return f"<Profile @{self.username}>"


class Post(Base):
    """Instagram post data"""
    __tablename__ = 'posts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey('profiles.id'), nullable=False)
    profile = relationship('Profile', back_populates='posts')

    post_id = Column(String(255))  # Instagram post ID
    shortcode = Column(String(50))  # e.g., "CUHYX5fqpf8"
    caption = Column(Text)
    caption_clean = Column(Text)  # Preprocessed caption
    hashtags = Column(JSON)  # List of hashtags
    post_type = Column(String(50))  # Image, Video, Sidecar
    likes_count = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    posted_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<Post {self.shortcode} by @{self.profile.username if self.profile else '?'}>"


class Cluster(Base):
    """Topic cluster from BERTopic"""
    __tablename__ = 'clusters'

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, nullable=False)  # BERTopic topic number
    name = Column(String(255))  # Human-readable name
    keywords = Column(JSON)  # List of top keywords
    size = Column(Integer, default=0)  # Number of profiles
    size_percent = Column(Float, default=0.0)

    # Relations
    profiles = relationship('Profile', back_populates='cluster')
    analysis = relationship('SegmentAnalysis', back_populates='cluster', uselist=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Cluster {self.topic_id}: {self.name}>"


class SegmentAnalysis(Base):
    """GPT-4o-mini analysis of cluster pain points"""
    __tablename__ = 'segment_analyses'

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(Integer, ForeignKey('clusters.id'), nullable=False, unique=True)
    cluster = relationship('Cluster', back_populates='analysis')

    # Pain point analysis
    segment_name = Column(String(255))  # e.g., "Застенчивые профессионалы"
    portrait = Column(Text)  # Портрет типичного представителя
    main_pain = Column(Text)  # Основная боль
    triggers = Column(JSON)  # Триггерные ситуации (list)
    desired_outcome = Column(Text)  # Желаемый результат
    client_phrase = Column(Text)  # Типичная формулировка клиента
    content_interests = Column(JSON)  # Какой контент потребляют (list)

    # Raw GPT response
    raw_response = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SegmentAnalysis {self.segment_name}>"


class FetchRun(Base):
    """Track Apify fetch runs"""
    __tablename__ = 'fetch_runs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(255))  # Apify run ID
    target_profile = Column(String(255))
    status = Column(String(50))  # pending, running, completed, failed
    profiles_fetched = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    def __repr__(self):
        return f"<FetchRun {self.run_id} - {self.status}>"
