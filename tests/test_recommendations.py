from abstracts_explorer.recommendations import personalized_pagerank, recommend_papers


class FakeCollection:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def get(self, ids=None, include=None):
        ids = ids or list(self.embeddings)
        found_ids = [uid for uid in ids if uid in self.embeddings]
        return {
            "ids": found_ids,
            "embeddings": [self.embeddings[uid] for uid in found_ids],
        }


class FakeEmbeddingsManager:
    def __init__(self, embeddings, query_embedding=None):
        self.collection = FakeCollection(embeddings)
        self.query_embedding = query_embedding or [1.0, 0.0]

    def generate_embedding(self, text):
        return self.query_embedding


class FakeDatabase:
    def __init__(self, papers):
        self.papers = papers

    def search_papers(
        self,
        keyword=None,
        field_filters=None,
        sessions=None,
        years=None,
        conferences=None,
        limit=100,
        **kwargs,
    ):
        results = list(self.papers)

        if sessions:
            results = [paper for paper in results if paper.get("session") in sessions]
        if years:
            results = [paper for paper in results if paper.get("year") in years]
        if conferences:
            results = [paper for paper in results if paper.get("conference") in conferences]
        if field_filters:
            for field, value in field_filters.items():
                needle = str(value).lower()
                filtered = []
                for paper in results:
                    field_value = paper.get(field)
                    if isinstance(field_value, list):
                        haystack = " ".join(str(item) for item in field_value).lower()
                    else:
                        haystack = str(field_value or "").lower()
                    if needle in haystack:
                        filtered.append(paper)
                results = filtered
        if keyword:
            needle = keyword.lower()
            results = [
                paper
                for paper in results
                if needle in (paper.get("title") or "").lower()
                or needle in (paper.get("abstract") or "").lower()
                or needle in " ".join(paper.get("keywords") or []).lower()
            ]

        if limit:
            return results[:limit]
        return results


def test_personalized_pagerank_biases_toward_seed_neighborhood():
    graph = {
        "seed": {"near": 1.0},
        "near": {"seed": 1.0, "far": 0.1},
        "far": {"near": 0.1},
    }

    ranks = personalized_pagerank(graph, {"seed": 1.0})

    assert ranks["near"] > ranks["far"]
    assert ranks["seed"] > 0


def test_recommend_papers_excludes_direct_author_seed_and_returns_coauthor_neighbor():
    papers = [
        {
            "uid": "p1",
            "title": "Alice and Bob seed paper",
            "authors": ["Alice", "Bob"],
            "abstract": "Seed work about scientific agents.",
            "keywords": ["agents"],
            "session": "Agents",
            "year": 2026,
            "conference": "HAICON",
        },
        {
            "uid": "p2",
            "title": "Bob follow-up paper",
            "authors": ["Bob"],
            "abstract": "Related work by a coauthor.",
            "keywords": ["agents"],
            "session": "Agents",
            "year": 2026,
            "conference": "HAICON",
        },
        {
            "uid": "p3",
            "title": "Distant paper",
            "authors": ["Carol"],
            "abstract": "A less connected topic.",
            "keywords": ["biology"],
            "session": "Bio",
            "year": 2026,
            "conference": "HAICON",
        },
    ]
    embeddings = {
        "p1": [1.0, 0.0],
        "p2": [0.95, 0.05],
        "p3": [0.0, 1.0],
    }

    result = recommend_papers(
        database=FakeDatabase(papers),
        embeddings_manager=FakeEmbeddingsManager(embeddings),
        query='authors:"Alice"',
        conferences=["HAICON"],
        years=[2026],
        limit=2,
    )

    returned_ids = [paper["uid"] for paper in result["papers"]]
    assert "p1" not in returned_ids
    assert returned_ids[0] == "p2"
    assert result["direct_seed_count"] == 1
    assert result["algorithm"] == "personalized_pagerank"


def test_recommend_papers_falls_back_to_metadata_graph_without_embeddings():
    papers = [
        {
            "uid": "p1",
            "title": "Alice and Bob seed paper",
            "authors": ["Alice", "Bob"],
            "abstract": "Seed work about scientific agents.",
            "keywords": ["agents"],
            "session": "Agents",
            "year": 2026,
            "conference": "HAICON",
        },
        {
            "uid": "p2",
            "title": "Bob follow-up paper",
            "authors": ["Bob"],
            "abstract": "Related work by a coauthor.",
            "keywords": ["agents"],
            "session": "Agents",
            "year": 2026,
            "conference": "HAICON",
        },
        {
            "uid": "p3",
            "title": "Distant paper",
            "authors": ["Carol"],
            "abstract": "A less connected topic.",
            "keywords": ["biology"],
            "session": "Bio",
            "year": 2026,
            "conference": "HAICON",
        },
    ]

    result = recommend_papers(
        database=FakeDatabase(papers),
        embeddings_manager=FakeEmbeddingsManager({}),
        query='authors:"Alice"',
        conferences=["HAICON"],
        years=[2026],
        limit=2,
    )

    returned_ids = [paper["uid"] for paper in result["papers"]]
    assert "p1" not in returned_ids
    assert returned_ids[0] == "p2"
    assert result["embedded_candidate_count"] == 0
    assert result["recommendation_mode"] == "metadata_graph"
    assert "metadata-only PageRank" in result["warning"]
