# -*- coding: utf-8 -*-
"""
tests/test_threads_bulk_crawler.py: Unit tests for bulk search and profile crawler in ThreadsHarvester.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.scrapers.threads_harvester import (
    search_threads_posts,
    harvest_threads_user_posts,
    harvest_threads_bulk,
    harvest_threads_search,
    ThreadsHarvestResult,
)


MOCK_SEARCH_HTML = """
<!DOCTYPE html>
<html>
<body>
    <a href="/@user1/post/C_post001">Post 1</a>
    <a href="/@user2/post/C_post002">Post 2</a>
    <a href="/t/C_post003">Post 3</a>
    <a href="/@user1/post/C_post001">Duplicate Post 1</a>
</body>
</html>
"""

MOCK_USER_HTML = """
<!DOCTYPE html>
<html>
<body>
    <a href="/@kz_creator/post/P_post101">Creator Post 1</a>
    <a href="/@kz_creator/post/P_post102">Creator Post 2</a>
</body>
</html>
"""

MOCK_POST_HTML_1 = """
<!DOCTYPE html>
<html>
<head>
    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "SocialMediaPosting",
        "identifier": "post_1",
        "articleBody": "Бүгін каспиден жаңа кітап сатып алдым, керемет.",
        "comment": [
            {
                "@type": "Comment",
                "identifier": "c_1",
                "text": "Өте жақсы таңдау, құттықтаймын!"
            }
        ]
    }
    </script>
</head>
</html>
"""

MOCK_POST_HTML_2 = """
<!DOCTYPE html>
<html>
<head>
    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "SocialMediaPosting",
        "identifier": "post_2",
        "articleBody": "Здравствуйте, подскажите где заказать доставку?",
        "comment": [
            {
                "@type": "Comment",
                "identifier": "c_2",
                "text": "В наличии нет сейчас, закончились."
            }
        ]
    }
    </script>
</head>
</html>
"""


class TestThreadsBulkCrawler(unittest.TestCase):
    @patch("httpx.Client.get")
    def test_search_threads_posts_extracts_unique_urls(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = MOCK_SEARCH_HTML
        mock_get.return_value = mock_resp

        urls = search_threads_posts("қазақша", limit=5)
        self.assertEqual(len(urls), 3)
        self.assertIn("https://www.threads.net/t/C_post001", urls)
        self.assertIn("https://www.threads.net/t/C_post002", urls)
        self.assertIn("https://www.threads.net/t/C_post003", urls)

    @patch("httpx.Client.get")
    def test_harvest_threads_user_posts(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = MOCK_USER_HTML
        mock_get.return_value = mock_resp

        urls = harvest_threads_user_posts("kz_creator", limit=5)
        self.assertEqual(len(urls), 2)
        self.assertIn("https://www.threads.net/t/P_post101", urls)
        self.assertIn("https://www.threads.net/t/P_post102", urls)

    @patch("httpx.Client.get")
    def test_harvest_threads_bulk_aggregates_and_filters(self, mock_get):
        # Return mock html for post 1, then post 2
        resp1 = MagicMock(status_code=200, text=MOCK_POST_HTML_1)
        resp2 = MagicMock(status_code=200, text=MOCK_POST_HTML_2)
        mock_get.side_effect = [resp1, resp2]

        with tempfile.TemporaryDirectory() as tmpdir:
            urls = [
                "https://www.threads.net/t/post_1",
                "https://www.threads.net/t/post_2",
            ]
            result = harvest_threads_bulk(
                urls=urls,
                output_dir=tmpdir,
                filter_non_kazakh=True,
            )

            self.assertIsInstance(result, ThreadsHarvestResult)
            self.assertTrue(result.success)
            self.assertTrue(os.path.isfile(result.batch_file_path))

            with open(result.batch_file_path, "r", encoding="utf-8") as f:
                batch_data = json.load(f)

            # Check combined items
            items = batch_data["items"]
            self.assertEqual(len(items), 4)

            # Kazakh items are active
            kk_items = [it for it in items if it["lang"] == "kk"]
            self.assertEqual(len(kk_items), 2)
            self.assertTrue(all(not it["discarded"] for it in kk_items))

            # Russian items are auto-discarded
            ru_items = [it for it in items if it["lang"] == "ru"]
            self.assertEqual(len(ru_items), 2)
            self.assertTrue(all(it["discarded"] for it in ru_items))


if __name__ == "__main__":
    unittest.main()
