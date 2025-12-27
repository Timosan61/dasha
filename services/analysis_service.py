"""
Analysis service using GPT-4o-mini for extracting pain points from clusters.
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

load_dotenv()

console = Console()

ANALYSIS_PROMPT = """Контекст: подписчики блога про коммуникации, пришли после рилса "как отвечать на комплименты".

Проанализируй данные Instagram-подписчиков из одного сегмента:

## Bio профилей (описания):
{bios}

## Примеры постов из аккаунтов сегмента:
{captions}

## Популярные хештеги в сегменте:
{hashtags}

## Ключевые слова сегмента: {keywords}

Задача: определи для этого сегмента:

1. **Название сегмента** (2-4 слова, описывающие типаж)
2. **Портрет представителя** (краткое описание типичного человека из сегмента)
3. **Основная боль в коммуникации** (главная проблема этих людей)
4. **Триггерные ситуации** (3-5 конкретных ситуаций, когда боль обостряется)
5. **Желаемый результат** (чего хотят достичь)
6. **Типичная формулировка клиента** (как они сами описывают проблему, в кавычках)
7. **Какой контент потребляют** (темы, которые им интересны, исходя из постов)

Ответ дай в формате JSON:
{{
    "segment_name": "...",
    "portrait": "...",
    "main_pain": "...",
    "triggers": ["...", "...", "..."],
    "desired_outcome": "...",
    "client_phrase": "...",
    "content_interests": ["...", "...", "..."]
}}"""


@dataclass
class SegmentAnalysisResult:
    cluster_id: int
    segment_name: str
    portrait: str
    main_pain: str
    triggers: List[str]
    desired_outcome: str
    client_phrase: str
    content_interests: List[str]
    raw_response: str


class AnalysisService:
    """Service for GPT-4o-mini analysis of audience segments"""

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def analyze_cluster(
        self,
        cluster_id: int,
        bios: List[str],
        keywords: List[str],
        captions: List[str] = None,
        hashtags: List[str] = None,
        max_bio_samples: int = 30,
        max_caption_samples: int = 50
    ) -> Optional[SegmentAnalysisResult]:
        """
        Analyze a single cluster to extract pain points.

        Args:
            cluster_id: Topic/cluster ID
            bios: List of bios in this cluster
            keywords: Top keywords for this cluster
            captions: List of post captions from this cluster
            hashtags: List of hashtags used in this cluster
            max_bio_samples: Maximum bio samples to send to GPT
            max_caption_samples: Maximum caption samples to send to GPT

        Returns:
            SegmentAnalysisResult or None on error
        """
        captions = captions or []
        hashtags = hashtags or []

        # Sample bios if too many
        sample_bios = bios[:max_bio_samples]
        sample_captions = captions[:max_caption_samples]

        # Format bios as numbered list
        bios_text = "\n".join([f"{i+1}. {bio}" for i, bio in enumerate(sample_bios)])

        # Format captions
        if sample_captions:
            captions_text = "\n".join([f"{i+1}. {cap[:200]}..." if len(cap) > 200 else f"{i+1}. {cap}"
                                       for i, cap in enumerate(sample_captions)])
        else:
            captions_text = "(нет данных о постах)"

        # Format hashtags (top 20)
        unique_hashtags = list(set(hashtags))[:20]
        hashtags_text = ", ".join([f"#{h}" for h in unique_hashtags]) if unique_hashtags else "(нет хештегов)"

        keywords_text = ", ".join(keywords[:10])

        prompt = ANALYSIS_PROMPT.format(
            bios=bios_text,
            captions=captions_text,
            hashtags=hashtags_text,
            keywords=keywords_text
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты эксперт по анализу целевой аудитории и маркетинговым исследованиям. Отвечай только на русском языке."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)

            return SegmentAnalysisResult(
                cluster_id=cluster_id,
                segment_name=data.get("segment_name", f"Сегмент {cluster_id}"),
                portrait=data.get("portrait", ""),
                main_pain=data.get("main_pain", ""),
                triggers=data.get("triggers", []),
                desired_outcome=data.get("desired_outcome", ""),
                client_phrase=data.get("client_phrase", ""),
                content_interests=data.get("content_interests", []),
                raw_response=raw
            )

        except json.JSONDecodeError as e:
            console.print(f"[red]JSON parse error for cluster {cluster_id}: {e}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]API error for cluster {cluster_id}: {e}[/red]")
            return None

    def analyze_all_clusters(
        self,
        clusters: List[Dict],  # [{id, bios, keywords, captions, hashtags}, ...]
    ) -> List[SegmentAnalysisResult]:
        """
        Analyze all clusters and return results.

        Args:
            clusters: List of cluster dicts with id, bios, keywords, captions, hashtags

        Returns:
            List of SegmentAnalysisResult
        """
        results = []

        for cluster in track(clusters, description="Analyzing clusters with GPT..."):
            result = self.analyze_cluster(
                cluster_id=cluster['id'],
                bios=cluster['bios'],
                keywords=cluster['keywords'],
                captions=cluster.get('captions', []),
                hashtags=cluster.get('hashtags', [])
            )
            if result:
                results.append(result)
                console.print(f"[green]Cluster {cluster['id']}: {result.segment_name}[/green]")

        return results

    def format_report(self, results: List[SegmentAnalysisResult]) -> str:
        """Format analysis results as markdown report"""
        lines = ["# Анализ сегментов аудитории\n"]

        for r in results:
            lines.append(f"## {r.segment_name}\n")
            if r.portrait:
                lines.append(f"**Портрет:** {r.portrait}\n")
            lines.append(f"**Основная боль:** {r.main_pain}\n")
            lines.append(f"**Триггерные ситуации:**")
            for t in r.triggers:
                lines.append(f"- {t}")
            lines.append(f"\n**Желаемый результат:** {r.desired_outcome}\n")
            lines.append(f"**Типичная фраза клиента:** _{r.client_phrase}_\n")
            if r.content_interests:
                lines.append(f"**Интересный контент:**")
                for c in r.content_interests:
                    lines.append(f"- {c}")
                lines.append("")
            lines.append("---\n")

        return "\n".join(lines)


if __name__ == "__main__":
    # Test
    service = AnalysisService()

    test_clusters = [
        {
            "id": 0,
            "keywords": ["психолог", "консультации", "онлайн", "помощь"],
            "bios": [
                "психолог помогаю найти себя консультации онлайн",
                "практикующий психолог работаю с тревожностью",
                "психотерапевт клинический психолог",
                "психолог консультации семейные пары",
                "детский психолог игровая терапия",
            ]
        }
    ]

    results = service.analyze_all_clusters(test_clusters)
    if results:
        report = service.format_report(results)
        console.print(report)
