from convtts_slr.protocol import SearchConfig
from convtts_slr.source import ArxivSource, Source


def main(source: Source):
    # Example usage of the ArxivSource and CitationSource classes
    config = SearchConfig(
        blocks=[["machine learning", "deep learning"], ["speech recognition"]],
        from_year=2020,
        to_date="2023-12-31",
        max_results_per_source=10,
    )
    results = source.search(config)
    for result in results:
        print(f"Title: {result.title}, Authors: {', '.join(result.authors)}, Year: {result.year}")


if __name__ == "__main__":
    # Example usage of the ArxivSource and CitationSource classes
    arxiv_source = ArxivSource()
    main(arxiv_source)
