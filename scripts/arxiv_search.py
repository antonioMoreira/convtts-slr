from whenever import Instant

from convtts_slr.protocol import SearchConfig
from convtts_slr.source import ArxivSource, Source


def main(source: Source):
    config = SearchConfig(
        blocks=[["machine learning", "deep learning"], ["speech recognition"]],
        from_year=2025,
        to_date=Instant.now().to_tz("UTC").date(),
        max_results_per_source=5,
    )
    result = source.search(config)
    print(f"Query: {result.query}")
    for paper in result.papers:
        print(f"Title: {paper.title}, Year: {paper.year}, arXiv: {paper.arxiv_id}")


if __name__ == "__main__":
    main(ArxivSource())
