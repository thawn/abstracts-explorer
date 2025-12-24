#!/usr/bin/env python
"""
Script to extract a diverse subset of NeurIPS 2025 papers for testing.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent / "src"))

from neurips_abstracts.downloader import download_neurips_data


def main():
    print("Downloading NeurIPS 2025 data...")
    data = download_neurips_data(year=2025)

    results = data.get("results", [])
    print(f"Total papers: {len(results)}")

    # Select diverse subset:
    # 1. Paper with no keywords
    # 2. Paper with keywords
    # 3. Paper with single author
    # 4. Paper with many authors
    # 5. Oral presentation
    # 6. Poster presentation
    # 7. Spotlight presentation
    # 8. Different topics

    subset = []

    # Find paper with no keywords
    for paper in results:
        if not paper.get("keywords") or len(paper.get("keywords", [])) == 0:
            subset.append(paper)
            print(f"\n1. No keywords: {paper['name'][:60]}")
            print(f"   Authors: {len(paper.get('authors', []))}")
            print(f"   Decision: {paper.get('decision')}")
            break

    # Find paper with keywords
    for paper in results:
        if paper.get("keywords") and len(paper.get("keywords", [])) > 0:
            subset.append(paper)
            print(f"\n2. With keywords: {paper['name'][:60]}")
            print(f"   Keywords: {paper.get('keywords')}")
            print(f"   Authors: {len(paper.get('authors', []))}")
            break

    # Find paper with single author
    for paper in results:
        if len(paper.get("authors", [])) == 1:
            subset.append(paper)
            print(f"\n3. Single author: {paper['name'][:60]}")
            print(f"   Author: {paper['authors'][0].get('fullname')}")
            print(f"   Decision: {paper.get('decision')}")
            break

    # Find paper with many authors (7+)
    for paper in results:
        if len(paper.get("authors", [])) >= 7:
            subset.append(paper)
            print(f"\n4. Many authors ({len(paper['authors'])}): {paper['name'][:60]}")
            print(f"   Decision: {paper.get('decision')}")
            break

    # Find oral presentation
    for paper in results:
        if paper.get("decision") == "Accept (oral)":
            subset.append(paper)
            print(f"\n5. Oral: {paper['name'][:60]}")
            topic = paper.get("topic") or "N/A"
            print(f"   Topic: {topic[:50]}")
            break

    # Find poster presentation
    for paper in results:
        if paper.get("decision") == "Accept (poster)":
            subset.append(paper)
            print(f"\n6. Poster: {paper['name'][:60]}")
            topic = paper.get("topic") or "N/A"
            print(f"   Topic: {topic[:50]}")
            break

    # Find spotlight presentation
    for paper in results:
        if paper.get("decision") == "Accept (spotlight)":
            subset.append(paper)
            print(f"\n7. Spotlight: {paper['name'][:60]}")
            topic = paper.get("topic") or "N/A"
            print(f"   Topic: {topic[:50]}")
            break

    # Find paper with no institution for author
    for paper in results:
        authors = paper.get("authors", [])
        if any(not author.get("institution") for author in authors):
            subset.append(paper)
            print(f"\n8. Author without institution: {paper['name'][:60]}")
            print(f"   Authors: {len(authors)}")
            break

    print(f"\n\nTotal subset papers: {len(subset)}")

    # Save subset
    output = {"count": len(subset), "results": subset}

    with open("test_subset.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nSubset saved to test_subset.json")

    # Print Python dict for copying
    print("\n" + "=" * 80)
    print("Python dict for test (first 5 papers):")
    print("=" * 80)
    for i, paper in enumerate(subset[:5], 1):
        print(f"\n# Paper {i}: {paper['name'][:60]}")
        print(
            f"# Authors: {len(paper.get('authors', []))}, Keywords: {len(paper.get('keywords', []))}, Decision: {paper.get('decision')}"
        )


if __name__ == "__main__":
    main()
