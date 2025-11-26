"""
Integration tests for the neurips_abstracts package.

Note: This test file contains integration tests for the new schema with integer IDs
and proper author relationships. Tests using the old schema have been removed.
See test_authors.py for comprehensive tests of the new schema.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from neurips_abstracts import download_json, DatabaseManager
from neurips_abstracts.downloader import download_neurips_data

# Fixtures imported from conftest.py:
# - sample_neurips_data: List of 2 papers with authors
# - mock_response: Mock HTTP response with sample data


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_download_and_load_workflow(self, tmp_path, mock_response, sample_neurips_data):
        """Test the complete workflow: download and load into database."""
        # Setup paths
        json_file = tmp_path / "neurips_data.json"
        db_file = tmp_path / "neurips.db"

        # Step 1: Download JSON
        with patch("neurips_abstracts.downloader.requests.get", return_value=mock_response):
            data = download_json(
                "https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json", output_path=json_file
            )

        assert data == sample_neurips_data
        assert json_file.exists()

        # Step 2: Load into database (data is a list from conftest.py)
        with DatabaseManager(db_file) as db:
            db.create_tables()
            count = db.load_json_data(data)

            assert count == 2
            assert db.get_paper_count() == 2

            # Step 3: Query the data by eventtype (new schema)
            # Note: conftest.py sample data has "Poster" and "Oral" papers
            poster_papers = db.search_papers(eventtype="Poster")
            assert len(poster_papers) == 1
            assert poster_papers[0]["name"] == "Deep Learning with Neural Networks"

            oral_papers = db.search_papers(eventtype="Oral")
            assert len(oral_papers) == 1
            assert oral_papers[0]["name"] == "Advances in Computer Vision"

    def test_download_neurips_and_load(self, tmp_path, mock_response, sample_neurips_data):
        """Test using the convenience function for NeurIPS data."""
        db_file = tmp_path / "neurips.db"

        # Download NeurIPS data
        with patch("neurips_abstracts.downloader.requests.get", return_value=mock_response):
            data = download_neurips_data(year=2025)

        # Load into database (data is a list from conftest.py)
        with DatabaseManager(db_file) as db:
            db.create_tables()
            count = db.load_json_data(data)

            assert count == 2

            # Search for papers - both papers have "neural" or "vision" keywords
            results = db.search_papers(keyword="neural")
            assert len(results) >= 1  # At least one paper matches

    def test_empty_database_queries(self, tmp_path):
        """Test querying an empty database."""
        db_file = tmp_path / "empty.db"

        with DatabaseManager(db_file) as db:
            db.create_tables()

            assert db.get_paper_count() == 0

            results = db.search_papers(keyword="anything")
            assert len(results) == 0

            results = db.search_papers(eventtype="Oral")
            assert len(results) == 0

    def test_database_persistence(self, tmp_path, mock_response, sample_neurips_data):
        """Test that database persists across connections."""
        db_file = tmp_path / "persistent.db"

        # Load data in first connection
        with patch("neurips_abstracts.downloader.requests.get", return_value=mock_response):
            data = download_neurips_data()

        with DatabaseManager(db_file) as db:
            db.create_tables()
            db.load_json_data(data)

        # Query in second connection
        with DatabaseManager(db_file) as db:
            assert db.get_paper_count() == 2

            # Search for "deep" which appears in first paper's keywords and title
            results = db.search_papers(keyword="deep")
            assert len(results) == 1

            # Search for papers - both papers exist
            all_results = db.search_papers()
            assert len(all_results) == 2

    def _get_real_neurips_subset(self):
        """
        Get the real NeurIPS 2025 subset data used for testing.

        Returns
        -------
        dict
            Dictionary with 'papers' key containing list of 7 papers.

        Notes
        -----
        This is a helper method to avoid code duplication between
        test_real_neurips_data_subset and test_embeddings_end_to_end_with_real_data.
        """
        return {
            "count": 7,
            "results": [
                # Paper 1: No keywords, 7 authors, poster
                {
                    "id": 119718,
                    "uid": "bad5f33780c42f2588878a9d07405083",
                    "name": "Coloring Learning for Heterophilic Graph Representation",
                    "authors": [
                        {
                            "id": 457880,
                            "fullname": "Miaomiao Huang",
                            "url": "http://neurips.cc/api/miniconf/users/457880?format=json",
                            "institution": "Northeastern University",
                        },
                        {
                            "id": 229953,
                            "fullname": "Yuhai Zhao",
                            "url": "http://neurips.cc/api/miniconf/users/229953?format=json",
                            "institution": "Northeastern University",
                        },
                        {
                            "id": 229927,
                            "fullname": "Daniel Zhengkui Wang",
                            "url": "http://neurips.cc/api/miniconf/users/229927?format=json",
                            "institution": "Singapore Institute of Technology",
                        },
                        {
                            "id": 220779,
                            "fullname": "Fenglong Ma",
                            "url": "http://neurips.cc/api/miniconf/users/220779?format=json",
                            "institution": "Pennsylvania State University",
                        },
                        {
                            "id": 457890,
                            "fullname": "Yejiang Wang",
                            "url": "http://neurips.cc/api/miniconf/users/457890?format=json",
                            "institution": "Northeastern University",
                        },
                        {
                            "id": 457879,
                            "fullname": "Meixia Wang",
                            "url": "http://neurips.cc/api/miniconf/users/457879?format=json",
                            "institution": "Northeastern University",
                        },
                        {
                            "id": 448027,
                            "fullname": "Xingwei Wang",
                            "url": "http://neurips.cc/api/miniconf/users/448027?format=json",
                            "institution": "Northeastern University",
                        },
                    ],
                    "abstract": "Graph self-supervised learning aims to learn the intrinsic graph representations from unlabeled data.",
                    "topic": "General Machine Learning->Representation Learning",
                    "keywords": [],
                    "decision": "Accept (poster)",
                    "session": "San Diego Poster Session 6",
                    "eventtype": "Poster",
                    "event_type": "{location} Poster",
                    "room_name": "Exhibit Hall C,D,E",
                    "virtualsite_url": "/virtual/2025/poster/119718",
                    "url": None,
                    "sourceid": 1379,
                    "sourceurl": "https://openreview.net/group?id=NeurIPS.cc/2025/Conference",
                    "starttime": "2025-12-05T16:30:00-08:00",
                    "endtime": "2025-12-05T19:30:00-08:00",
                    "paper_url": "https://openreview.net/forum?id=7HVADbW8fh",
                    "poster_position": "#2504",
                },
                # Paper 2: Single author
                {
                    "id": 119663,
                    "uid": "20b02dc95171540bc52912baf3aa709d",
                    "name": "Miss-ReID: Delivering Robust Multi-Modality Object Re-Identification Despite Missing Modalities",
                    "authors": [
                        {
                            "id": 429076,
                            "fullname": "Xi ruida",
                            "url": "http://neurips.cc/api/miniconf/users/429076?format=json",
                            "institution": "State Key Laboratory of Electromechanical Integrated Manufacturing of High-Performance Electronic Equipment, Xidian University",
                        }
                    ],
                    "abstract": "Multi-modality object re-identification (ReID) has achieved remarkable progress.",
                    "topic": "Computer Vision->Other",
                    "keywords": [],
                    "decision": "Accept (poster)",
                    "session": "San Diego Poster Session 6",
                    "eventtype": "Poster",
                    "event_type": "{location} Poster",
                    "room_name": "Exhibit Hall C,D,E",
                    "virtualsite_url": "/virtual/2025/poster/119663",
                    "url": None,
                    "sourceid": 1379,
                    "sourceurl": "https://openreview.net/group?id=NeurIPS.cc/2025/Conference",
                    "starttime": "2025-12-05T16:30:00-08:00",
                    "endtime": "2025-12-05T19:30:00-08:00",
                    "paper_url": "https://openreview.net/forum?id=1tzlbKySxX",
                    "poster_position": "#2449",
                },
                # Paper 3: Oral presentation, 9 authors, null topic
                {
                    "id": 114995,
                    "uid": "f2a2c68d6bfdc87c8097f5b45e8c9d80",
                    "name": "DisCO: Reinforcing Large Reasoning Models with Discriminative Constraints",
                    "authors": [
                        {
                            "id": 441943,
                            "fullname": "Gang Li",
                            "url": "http://neurips.cc/api/miniconf/users/441943?format=json",
                            "institution": "Texas A&amp;M University",
                        },
                        {
                            "id": 296746,
                            "fullname": "Ming Lin",
                            "url": "http://neurips.cc/api/miniconf/users/296746?format=json",
                            "institution": "Oracle Cloud Infrastructure",
                        },
                        {
                            "id": 346388,
                            "fullname": "Tomer Galanti",
                            "url": "http://neurips.cc/api/miniconf/users/346388?format=json",
                            "institution": "Texas A&amp;M University",
                        },
                        {
                            "id": 429803,
                            "fullname": "Zhen Guo",
                            "url": "http://neurips.cc/api/miniconf/users/429803?format=json",
                            "institution": "Oracle Cloud Infrastructure",
                        },
                        {
                            "id": 454031,
                            "fullname": "Yilun Du",
                            "url": "http://neurips.cc/api/miniconf/users/454031?format=json",
                            "institution": "Harvard University",
                        },
                        {
                            "id": 439486,
                            "fullname": "Qiang Liu",
                            "url": "http://neurips.cc/api/miniconf/users/439486?format=json",
                            "institution": "UT Austin",
                        },
                        {
                            "id": 419761,
                            "fullname": "Shuiwang Ji",
                            "url": "http://neurips.cc/api/miniconf/users/419761?format=json",
                            "institution": "Texas A&amp;M University",
                        },
                        {
                            "id": 419767,
                            "fullname": "Tommi Jaakkola",
                            "url": "http://neurips.cc/api/miniconf/users/419767?format=json",
                            "institution": "MIT",
                        },
                        {
                            "id": 403093,
                            "fullname": "Zichao Yang",
                            "url": "http://neurips.cc/api/miniconf/users/403093?format=json",
                            "institution": "ByteDance",
                        },
                    ],
                    "abstract": "The recent success and openness of DeepSeek-R1 have brought widespread attention to Group Relative Policy Optimization (GRPO).",
                    "topic": None,
                    "keywords": [],
                    "decision": "Accept (oral)",
                    "session": "Oral Session 17",
                    "eventtype": "Oral",
                    "event_type": "{location} Oral",
                    "room_name": "Ballroom C",
                    "virtualsite_url": "/virtual/2025/oral/114995",
                    "url": None,
                    "sourceid": 1379,
                    "sourceurl": "https://openreview.net/group?id=NeurIPS.cc/2025/Conference",
                    "starttime": "2025-12-08T12:15:00-08:00",
                    "endtime": "2025-12-08T12:30:00-08:00",
                    "paper_url": "https://openreview.net/forum?id=1kPO9vCRbg",
                },
                # Paper 4: Spotlight presentation
                {
                    "id": 119969,
                    "uid": "c94d8a5e1d57a5b77a8be7e8c9d0e1a2",
                    "name": "Nonlinear Laplacians: Tunable principal component analysis using graph Laplacians",
                    "authors": [
                        {
                            "id": 413852,
                            "fullname": "Matthew Cho",
                            "url": "http://neurips.cc/api/miniconf/users/413852?format=json",
                            "institution": "MIT",
                        },
                        {
                            "id": 364144,
                            "fullname": "Nicolas Garcia Trillos",
                            "url": "http://neurips.cc/api/miniconf/users/364144?format=json",
                            "institution": "University of Wisconsin-Madison",
                        },
                    ],
                    "abstract": "We introduce a nonlinear analog of Laplacian-based principal component analysis.",
                    "topic": "Theory->Learning Theory",
                    "keywords": [],
                    "decision": "Accept (spotlight)",
                    "session": "Spotlight Session 8",
                    "eventtype": "Spotlight",
                    "event_type": "{location} Spotlight",
                    "room_name": "Ballroom A",
                    "virtualsite_url": "/virtual/2025/spotlight/119969",
                    "url": None,
                    "sourceid": 1379,
                    "sourceurl": "https://openreview.net/group?id=NeurIPS.cc/2025/Conference",
                    "starttime": "2025-12-06T14:30:00-08:00",
                    "endtime": "2025-12-06T14:45:00-08:00",
                    "paper_url": "https://openreview.net/forum?id=ABC123",
                },
                # Paper 5: Different poster with Computer Vision topic
                {
                    "id": 119801,
                    "uid": "e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6",
                    "name": "Vision-Language Models for Computer Vision Tasks",
                    "authors": [
                        {
                            "id": 450001,
                            "fullname": "John Doe",
                            "url": "http://neurips.cc/api/miniconf/users/450001?format=json",
                            "institution": "University of Example",
                        },
                        {
                            "id": 450002,
                            "fullname": "Jane Smith",
                            "url": "http://neurips.cc/api/miniconf/users/450002?format=json",
                            "institution": "",
                        },
                        {
                            "id": 450003,
                            "fullname": "Bob Johnson",
                            "url": "http://neurips.cc/api/miniconf/users/450003?format=json",
                            "institution": None,
                        },
                    ],
                    "abstract": "This is an example paper with authors that have missing institutions.",
                    "topic": "Applications->Vision",
                    "keywords": [],
                    "decision": "Accept (poster)",
                    "session": "Poster Session 1",
                    "eventtype": "Poster",
                    "event_type": "{location} Poster",
                    "room_name": "Exhibit Hall",
                    "virtualsite_url": "/virtual/2025/poster/119801",
                    "url": None,
                    "sourceid": 1379,
                    "sourceurl": "https://openreview.net/group?id=NeurIPS.cc/2025/Conference",
                    "starttime": "2025-12-05T14:00:00-08:00",
                    "endtime": "2025-12-05T16:00:00-08:00",
                    "paper_url": "https://openreview.net/forum?id=XYZ789",
                },
                # Paper 6: Paper with keywords (something different)
                {
                    "id": 119900,
                    "uid": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
                    "name": "Deep Reinforcement Learning with Transfer",
                    "authors": [
                        {
                            "id": 460001,
                            "fullname": "Alice Cooper",
                            "url": "http://neurips.cc/api/miniconf/users/460001?format=json",
                            "institution": "Stanford University",
                        },
                        {
                            "id": 460002,
                            "fullname": "Bob Dylan",
                            "url": "http://neurips.cc/api/miniconf/users/460002?format=json",
                            "institution": "MIT",
                        },
                    ],
                    "abstract": "We propose a novel approach to transfer learning in deep RL.",
                    "topic": "Reinforcement Learning->Deep RL",
                    "keywords": ["reinforcement learning", "transfer learning", "deep learning"],
                    "decision": "Accept (poster)",
                    "session": "Poster Session 3",
                    "eventtype": "Poster",
                    "event_type": "{location} Poster",
                    "room_name": "Exhibit Hall A",
                    "virtualsite_url": "/virtual/2025/poster/119900",
                    "url": None,
                    "sourceid": 1379,
                    "sourceurl": "https://openreview.net/group?id=NeurIPS.cc/2025/Conference",
                    "starttime": "2025-12-06T14:00:00-08:00",
                    "endtime": "2025-12-06T16:00:00-08:00",
                    "paper_url": "https://openreview.net/forum?id=ABC456",
                },
                # Paper 7: Another paper with 5 authors
                {
                    "id": 120000,
                    "uid": "f1e2d3c4b5a6f7e8d9c0b1a2f3e4d5c6",
                    "name": "Efficient Training of Large Language Models",
                    "authors": [
                        {
                            "id": 470001,
                            "fullname": "Emily Zhang",
                            "url": "http://neurips.cc/api/miniconf/users/470001?format=json",
                            "institution": "Google Research",
                        },
                        {
                            "id": 470002,
                            "fullname": "Michael Johnson",
                            "url": "http://neurips.cc/api/miniconf/users/470002?format=json",
                            "institution": "OpenAI",
                        },
                        {
                            "id": 470003,
                            "fullname": "Sarah Williams",
                            "url": "http://neurips.cc/api/miniconf/users/470003?format=json",
                            "institution": "DeepMind",
                        },
                        {
                            "id": 470004,
                            "fullname": "David Lee",
                            "url": "http://neurips.cc/api/miniconf/users/470004?format=json",
                            "institution": "Meta AI",
                        },
                        {
                            "id": 470005,
                            "fullname": "Lisa Chen",
                            "url": "http://neurips.cc/api/miniconf/users/470005?format=json",
                            "institution": "Anthropic",
                        },
                    ],
                    "abstract": "We present efficient techniques for training large language models at scale.",
                    "topic": "Deep Learning->Large Language Models",
                    "keywords": ["language models", "training efficiency", "optimization"],
                    "decision": "Accept (poster)",
                    "session": "Poster Session 4",
                    "eventtype": "Poster",
                    "event_type": "{location} Poster",
                    "room_name": "Exhibit Hall B",
                    "virtualsite_url": "/virtual/2025/poster/120000",
                    "url": None,
                    "sourceid": 1379,
                    "sourceurl": "https://openreview.net/group?id=NeurIPS.cc/2025/Conference",
                    "starttime": "2025-12-07T14:00:00-08:00",
                    "endtime": "2025-12-07T16:00:00-08:00",
                    "paper_url": "https://openreview.net/forum?id=DEF789",
                },
            ],
        }

    def test_real_neurips_data_subset(self, tmp_path):
        """
        Test with a diverse subset of actual NeurIPS 2025 data.

        This test uses 7 papers representing different use cases:
        1. Paper with no keywords (7 authors)
        2. Paper with single author
        3. Paper with many authors (9 authors)
        4. Oral presentation
        5. Poster presentation
        6. Spotlight presentation
        7. Paper with author missing institution
        """
        # Actual subset of NeurIPS 2025 data with diverse characteristics
        real_data = self._get_real_neurips_subset()

        db_file = tmp_path / "neurips_real_subset.db"

        with DatabaseManager(db_file) as db:
            # Create tables
            db.create_tables()

            # Load the real data subset
            count = db.load_json_data(real_data)
            assert count == 7, f"Expected 7 papers, got {count}"

            # Verify total counts
            assert db.get_paper_count() == 7

            # Test 1: Query by decision type
            oral_papers = db.search_papers(decision="Accept (oral)")
            assert len(oral_papers) == 1
            assert (
                oral_papers[0]["name"] == "DisCO: Reinforcing Large Reasoning Models with Discriminative Constraints"
            )

            poster_papers = db.search_papers(decision="Accept (poster)")
            assert len(poster_papers) == 5  # Papers 1, 2, 5, 6, 7

            spotlight_papers = db.search_papers(decision="Accept (spotlight)")
            assert len(spotlight_papers) == 1
            assert spotlight_papers[0]["name"].startswith("Nonlinear Laplacians")

            # Test 2: Query by event type
            oral_events = db.search_papers(eventtype="Oral")
            assert len(oral_events) == 1

            poster_events = db.search_papers(eventtype="Poster")
            assert len(poster_events) == 5  # Papers 1, 2, 5, 6, 7

            spotlight_events = db.search_papers(eventtype="Spotlight")
            assert len(spotlight_events) == 1

            # Test 3: Keyword search
            graph_papers = db.search_papers(keyword="graph")
            assert len(graph_papers) >= 1
            assert any("Graph" in p["name"] for p in graph_papers)

            learning_papers = db.search_papers(keyword="learning")
            assert len(learning_papers) >= 1

            # Test 4: Topic search
            cv_papers = db.search_papers(topic="Computer Vision")
            assert len(cv_papers) >= 1

            theory_papers = db.search_papers(topic="Theory")
            assert len(theory_papers) >= 1

            # Test 5: Verify author extraction and relationships
            # Check author count
            author_count = db.get_author_count()
            # 7 + 1 + 9 + 2 + 3 + 2 + 5 = 29 unique authors
            assert author_count == 29, f"Expected 29 unique authors, got {author_count}"

            # Test 6: Get authors for specific papers
            # Paper with 7 authors
            authors_paper1 = db.get_paper_authors(119718)
            assert len(authors_paper1) == 7
            assert authors_paper1[0]["fullname"] == "Miaomiao Huang"
            assert authors_paper1[0]["institution"] == "Northeastern University"

            # Paper with single author
            authors_paper2 = db.get_paper_authors(119663)
            assert len(authors_paper2) == 1
            assert authors_paper2[0]["fullname"] == "Xi ruida"

            # Paper with 9 authors (oral presentation)
            authors_paper3 = db.get_paper_authors(114995)
            assert len(authors_paper3) == 9
            assert authors_paper3[0]["fullname"] == "Gang Li"

            # Test 7: Search authors by institution
            northeastern_authors = db.search_authors(institution="Northeastern")
            assert len(northeastern_authors) >= 5  # At least 5 from first paper

            # Test 8: Search authors by name
            wang_authors = db.search_authors(name="Wang")
            assert len(wang_authors) >= 3  # Multiple Wang authors in first paper

            # Test 9: Get papers by author
            # Find an author and get their papers
            miaomiao = db.search_authors(name="Miaomiao Huang")
            assert len(miaomiao) == 1
            miaomiao_papers = db.get_author_papers(miaomiao[0]["id"])
            assert len(miaomiao_papers) >= 1

            # Test 10: Handle papers with null/empty fields
            # Check that paper with null topic is handled
            papers_with_null_topic = db.query("SELECT * FROM papers WHERE topic IS NULL")
            assert len(papers_with_null_topic) >= 1

            # Test 11: Verify empty keywords are handled correctly
            papers_with_no_keywords = db.query(
                "SELECT * FROM papers WHERE keywords = '' OR keywords IS NULL OR keywords = '[]'"
            )
            assert len(papers_with_no_keywords) == 5  # Papers 1-5 have no keywords

            # Test 11b: Verify papers with keywords
            papers_with_keywords = db.query(
                "SELECT * FROM papers WHERE keywords != '' AND keywords IS NOT NULL AND keywords != '[]'"
            )
            assert len(papers_with_keywords) == 2  # Papers 6 and 7 have keywords

            # Test 12: Verify authors with missing institutions
            authors_no_institution = db.query("SELECT * FROM authors WHERE institution IS NULL OR institution = ''")
            assert len(authors_no_institution) >= 2  # From paper 5

            # Test 13: Verify author order is preserved
            paper5_authors = db.get_paper_authors(119801)
            assert len(paper5_authors) == 3
            assert paper5_authors[0]["author_order"] == 1
            assert paper5_authors[1]["author_order"] == 2
            assert paper5_authors[2]["author_order"] == 3
            assert paper5_authors[0]["fullname"] == "John Doe"
            assert paper5_authors[1]["fullname"] == "Jane Smith"
            assert paper5_authors[2]["fullname"] == "Bob Johnson"

    def test_embeddings_end_to_end_with_real_data(self, tmp_path):
        """
        End-to-end test: Load real NeurIPS data, generate embeddings, and perform semantic search.

        This test verifies the complete embeddings workflow:
        1. Load papers from the real NeurIPS 2025 subset into database
        2. Generate embeddings for all papers with abstracts (requires LM Studio running)
        3. Perform semantic similarity searches
        4. Verify results are relevant and properly ranked
        5. Test metadata filtering in vector search

        Note: This test requires LM Studio to be running at http://localhost:1234.
        The test will be skipped if LM Studio is not available.
        """
        import pytest
        import requests
        from neurips_abstracts import DatabaseManager, EmbeddingsManager

        # Check if LM Studio is available before running the test
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=2)
            if response.status_code != 200:
                pytest.skip("LM Studio is not responding correctly")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pytest.skip("LM Studio is not available at http://localhost:1234")

        # Use the same real data subset from test_real_neurips_data_subset
        real_data = self._get_real_neurips_subset()

        db_file = tmp_path / "neurips_embeddings.db"
        chroma_path = tmp_path / "test_chroma_db"

        # Step 1: Load papers into database
        with DatabaseManager(db_file) as db:
            db.create_tables()
            count = db.load_json_data(real_data)
            assert count == 7
            assert db.get_paper_count() == 7

        # Step 2: Generate embeddings from database using real LM Studio API
        with EmbeddingsManager(chroma_path=chroma_path, collection_name="neurips_test") as em:
            em.create_collection()

            # Embed only papers with non-empty abstracts
            embedded_count = em.embed_from_database(
                db_file, batch_size=10, where_clause="abstract IS NOT NULL AND abstract != ''"
            )

            # All 7 papers should have abstracts
            assert embedded_count == 7

            # Verify collection stats
            stats = em.get_collection_stats()
            assert stats["name"] == "neurips_test"
            assert stats["count"] == 7

            # Step 3: Test semantic search - Find papers about "graph learning"
            graph_results = em.search_similar("graph learning representation", n_results=3)

            assert "ids" in graph_results
            assert len(graph_results["ids"][0]) <= 3
            assert len(graph_results["ids"][0]) > 0

            # Verify we got paper IDs back
            result_ids = [int(pid) for pid in graph_results["ids"][0]]
            assert all(pid in [119718, 119663, 114995, 119969, 119801, 119900, 120000] for pid in result_ids)

            # Verify distances are present and reasonable
            distances = graph_results["distances"][0]
            assert all(isinstance(d, (int, float)) for d in distances)
            assert len(distances) == len(result_ids)

            # Verify metadata is preserved
            metadatas = graph_results["metadatas"][0]
            assert all("title" in meta for meta in metadatas)
            assert all("decision" in meta for meta in metadatas)

            # Step 4: Test search with different query
            reasoning_results = em.search_similar("reasoning and language models", n_results=2)

            assert len(reasoning_results["ids"][0]) <= 2
            assert len(reasoning_results["ids"][0]) > 0

            # Step 5: Test metadata filtering - only poster papers
            poster_results = em.search_similar("machine learning", n_results=5, where={"decision": "Accept (poster)"})

            # Should return results
            assert len(poster_results["ids"][0]) > 0

            # Verify all returned papers are posters
            for meta in poster_results["metadatas"][0]:
                assert "poster" in meta["decision"].lower()

            # Step 6: Verify documents (abstracts) are returned
            docs = graph_results["documents"][0]
            assert all(isinstance(doc, str) for doc in docs)
            assert all(len(doc) > 0 for doc in docs)

            # Step 7: Test that embeddings persist across sessions
            # Close and reopen the manager
            em.close()

        # Reopen and verify data persists
        with EmbeddingsManager(chroma_path=chroma_path, collection_name="neurips_test") as em:
            em.create_collection()

            stats = em.get_collection_stats()
            assert stats["count"] == 7

            # Can still perform searches
            results = em.search_similar("neural networks", n_results=3)
            assert len(results["ids"][0]) > 0
