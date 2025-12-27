"""
Clustering service using OpenAI Embeddings API + sklearn.
Lightweight alternative to BERTopic - no PyTorch/CUDA required.
"""

import os
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import pickle
from collections import Counter

from openai import OpenAI
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import track

load_dotenv()

console = Console()

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


@dataclass
class ClusterResult:
    topic_id: int
    name: str
    keywords: List[str]
    size: int
    size_percent: float
    sample_docs: List[str] = field(default_factory=list)


class ClusteringService:
    """Service for clustering using OpenAI Embeddings + sklearn HDBSCAN"""

    def __init__(self, min_cluster_size: int = 15, use_kmeans: bool = False, n_clusters: int = 15):
        """
        Initialize clustering service.

        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            use_kmeans: Use KMeans instead of HDBSCAN (fixed number of clusters)
            n_clusters: Number of clusters for KMeans
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=api_key)
        self.min_cluster_size = min_cluster_size
        self.use_kmeans = use_kmeans
        self.n_clusters = n_clusters

        self.model_dir = Path(__file__).parent.parent / 'data' / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Store embeddings and labels after fit
        self.embeddings = None
        self.labels = None
        self.documents = None

    def get_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Get embeddings from OpenAI API.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            numpy array of embeddings
        """
        all_embeddings = []

        # Process in batches
        for i in track(range(0, len(texts), batch_size), description="Getting embeddings from OpenAI..."):
            batch = texts[i:i + batch_size]

            # Filter empty texts
            batch = [t if t else " " for t in batch]

            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def extract_keywords(self, documents: List[str], labels: np.ndarray, top_n: int = 10) -> dict:
        """
        Extract keywords for each cluster using TF-IDF.

        Args:
            documents: Original documents
            labels: Cluster labels
            top_n: Number of keywords per cluster

        Returns:
            Dict mapping cluster_id to list of keywords
        """
        keywords_per_cluster = {}

        # TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Russian stopwords would be handled in preprocessing
            ngram_range=(1, 2)
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError:
            # Not enough documents
            return {}

        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Skip outliers
                continue

            # Get indices of documents in this cluster
            cluster_indices = np.where(labels == label)[0]

            if len(cluster_indices) == 0:
                continue

            # Sum TF-IDF scores for cluster documents
            cluster_tfidf = tfidf_matrix[cluster_indices].sum(axis=0).A1

            # Get top keywords
            top_indices = cluster_tfidf.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices]

            keywords_per_cluster[label] = keywords

        return keywords_per_cluster

    def fit(self, documents: List[str]) -> Tuple[List[int], List[ClusterResult]]:
        """
        Cluster documents using OpenAI embeddings + HDBSCAN/KMeans.

        Args:
            documents: List of preprocessed texts

        Returns:
            Tuple of (topic assignments, cluster results)
        """
        # Filter empty documents
        valid_docs = [d for d in documents if d and len(d.strip()) > 5]
        console.print(f"[blue]Clustering {len(valid_docs)} documents...[/blue]")

        if len(valid_docs) < 10:
            console.print("[red]Error: Too few documents for clustering (need at least 10)[/red]")
            return [], []

        # Get embeddings from OpenAI
        self.embeddings = self.get_embeddings(valid_docs)
        self.documents = valid_docs

        # Cluster using HDBSCAN or KMeans
        if self.use_kmeans:
            console.print(f"[blue]Using KMeans with {self.n_clusters} clusters...[/blue]")
            clusterer = KMeans(
                n_clusters=min(self.n_clusters, len(valid_docs) // 5),
                random_state=42,
                n_init=10
            )
            self.labels = clusterer.fit_predict(self.embeddings)
        else:
            console.print(f"[blue]Using HDBSCAN (min_cluster_size={self.min_cluster_size})...[/blue]")
            clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=5,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            self.labels = clusterer.fit_predict(self.embeddings)

        # Extract keywords for each cluster
        keywords_per_cluster = self.extract_keywords(valid_docs, self.labels)

        # Build results
        results = []
        total_docs = len(valid_docs)
        label_counts = Counter(self.labels)

        for label, count in label_counts.items():
            if label == -1:  # Skip outliers
                console.print(f"[dim]Outliers: {count} documents[/dim]")
                continue

            keywords = keywords_per_cluster.get(label, [])

            # Get sample documents
            sample_indices = np.where(self.labels == label)[0][:5]
            sample_docs = [valid_docs[i] for i in sample_indices]

            results.append(ClusterResult(
                topic_id=int(label),
                name=f"Topic {label}: {', '.join(keywords[:3])}" if keywords else f"Topic {label}",
                keywords=keywords,
                size=count,
                size_percent=round(count / total_docs * 100, 1),
                sample_docs=sample_docs
            ))

        # Sort by size
        results.sort(key=lambda x: x.size, reverse=True)

        console.print(f"[green]Found {len(results)} clusters[/green]")
        self._print_summary(results)

        # Return labels for all original documents (map back)
        # Create mapping from valid_docs indices to labels
        all_labels = []
        valid_idx = 0
        for doc in documents:
            if doc and len(doc.strip()) > 5:
                all_labels.append(int(self.labels[valid_idx]))
                valid_idx += 1
            else:
                all_labels.append(-1)  # Outlier for empty docs

        return all_labels, results

    def _print_summary(self, results: List[ClusterResult]):
        """Print summary table of clusters"""
        table = Table(title="Cluster Summary")
        table.add_column("Topic", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("%", style="yellow")
        table.add_column("Keywords", style="white")

        for r in results[:10]:  # Top 10
            table.add_row(
                str(r.topic_id),
                str(r.size),
                f"{r.size_percent}%",
                ", ".join(r.keywords[:5])
            )

        console.print(table)

    def get_umap_coords(self) -> Tuple[List[float], List[float]]:
        """
        Reduce embeddings to 2D for visualization using simple PCA.
        (Avoids heavy UMAP dependency)
        """
        if self.embeddings is None:
            raise ValueError("Must call fit() first")

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(self.embeddings)

        return coords[:, 0].tolist(), coords[:, 1].tolist()

    def save_model(self, name: str = "clustering_model"):
        """Save embeddings and labels to disk"""
        filepath = self.model_dir / f"{name}.pkl"
        data = {
            'embeddings': self.embeddings,
            'labels': self.labels,
            'documents': self.documents
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        console.print(f"[dim]Model saved to {filepath}[/dim]")

    def load_model(self, name: str = "clustering_model") -> bool:
        """Load embeddings and labels from disk"""
        filepath = self.model_dir / f"{name}.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.labels = data['labels']
            self.documents = data['documents']
            console.print(f"[green]Model loaded from {filepath}[/green]")
            return True
        return False

    def assign_topic(self, document: str) -> Tuple[int, float]:
        """
        Assign topic to a new document based on nearest cluster centroid.
        """
        if self.embeddings is None or self.labels is None:
            raise ValueError("Must call fit() first")

        # Get embedding for new document
        embedding = self.get_embeddings([document])[0]

        # Find nearest cluster by comparing to cluster centroids
        unique_labels = set(self.labels) - {-1}

        best_label = -1
        best_distance = float('inf')

        for label in unique_labels:
            # Get centroid of this cluster
            cluster_indices = np.where(self.labels == label)[0]
            centroid = self.embeddings[cluster_indices].mean(axis=0)

            # Calculate distance
            distance = cosine_distances([embedding], [centroid])[0][0]

            if distance < best_distance:
                best_distance = distance
                best_label = label

        # Convert distance to similarity score (0-1)
        confidence = max(0, 1 - best_distance)

        return best_label, confidence


if __name__ == "__main__":
    # Test with sample data
    service = ClusteringService(min_cluster_size=3)

    test_docs = [
        "психолог консультации онлайн помощь терапия",
        "психотерапевт тревожность депрессия лечение",
        "коуч личностный рост развитие мотивация",
        "маркетолог smm продвижение реклама бренд",
        "таргетолог реклама instagram facebook",
        "мама дети семья воспитание",
        "мама двоих дочки сыновья",
        "путешествия travel мир страны",
        "фотограф съемка портрет свадьба",
        "предприниматель бизнес стартап инвестиции",
        "дизайнер интерьер ремонт стиль",
        "йога медитация практика здоровье",
        "фитнес спорт тренировки зож",
    ]

    topics, clusters = service.fit(test_docs)
    console.print(f"\n[bold]Topic assignments:[/bold] {topics}")
