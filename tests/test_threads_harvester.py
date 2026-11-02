# -*- coding: utf-8 -*-
"""
tests/test_threads_harvester.py: Unit tests for public Threads scraper and staging batch creator.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.scrapers.threads_harvester import (
    extract_comments_from_html,
    create_staging_batch,
    harvest_threads_url,
    ThreadsHarvestResult,
)


SAMPLE_THREADS_HTML_WITH_JSON_LD = """
<!DOCTYPE html>
<html>
<head>
    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "SocialMediaPosting",
        "identifier": "post_1001",
        "articleBody": "Бүгін Алматыда ауа райы керемет болып тұр @daulet_kz https://t.me/news",
        "comment": [
            {
                "@type": "Comment",
                "identifier": "th_c01",
                "text": "Иә расымен де каспиден тапсырыс беріп скрин жасадым +77015554433"
            },
            {
                "@type": "Comment",
                "identifier": "th_c02",
                "text": "Ватсапқа жібердіңіз бе @aigul_kz?"
            }
        ]
    }
    </script>
</head>
<body>
    <main>Threads Post</main>
</body>
</html>
"""

SAMPLE_THREADS_HTML_SSR_DATA_SJS = """
<!DOCTYPE html>
<html>
<head>
    <meta property="og:description" content="Басты тақырып бойынша талқылау @username">
    <script type="application/json" data-sjs>
    {
        "require": [
            ["ScheduledServerJS", "handle", [], [
                {
                    "__bbox": {
                        "result": {
                            "data": {
                                "data": {
                                    "containing_thread": {
                                        "thread_items": [
                                            {
                                                "post": {
                                                    "id": "post_2001",
                                                    "caption": {"text": "Негізгі жазба мәтіні https://threads.net"}
                                                }
                                            },
                                            {
                                                "post": {
                                                    "id": "post_2002",
                                                    "caption": {"text": "Каспилену процесі жүріп жатыр @kaspi_user"}
                                                }
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            ]]
        ]
    }
    </script>
</head>
<body></body>
</html>
"""


class TestThreadsHarvester(unittest.TestCase):
    def test_extract_comments_from_json_ld(self):
        comments = extract_comments_from_html(SAMPLE_THREADS_HTML_WITH_JSON_LD, default_post_id="post_1001")
        self.assertGreaterEqual(len(comments), 2)

        # First item is main post
        main_post = comments[0]
        self.assertEqual(main_post["post_id"], "post_1001")
        self.assertIn("@user_anon", main_post["text_anonymized"])
        self.assertIn("[URL]", main_post["text_anonymized"])
        self.assertNotIn("@daulet_kz", main_post["text_anonymized"])

        # Second item is first comment with phone number
        first_comment = comments[1]
        self.assertIn("[PHONE]", first_comment["text_anonymized"])
        self.assertNotIn("+77015554433", first_comment["text_anonymized"])

        # Third item is second comment with handle
        second_comment = comments[2]
        self.assertIn("@user_anon", second_comment["text_anonymized"])
        self.assertNotIn("@aigul_kz", second_comment["text_anonymized"])

    def test_extract_comments_from_ssr_data_sjs(self):
        comments = extract_comments_from_html(SAMPLE_THREADS_HTML_SSR_DATA_SJS, default_post_id="post_2001")
        self.assertGreaterEqual(len(comments), 2)
        texts = [c["text_anonymized"] for c in comments]
        self.assertTrue(any("Каспилену" in t or "каспилену" in t.lower() for t in texts))
        self.assertTrue(all("@kaspi_user" not in t for t in texts))

    def test_create_staging_batch_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_items = [
                {
                    "comment_id": "th_981273",
                    "post_id": "C_abc123",
                    "text": "Каспиден скрин жасап ватсапқа жібердім @user https://threads.net",
                }
            ]
            batch_path, batch_data = create_staging_batch(
                source_query="https://www.threads.net/@user/post/C_abc123",
                raw_items=raw_items,
                output_dir=tmpdir,
            )

            self.assertTrue(os.path.isfile(batch_path))
            self.assertEqual(batch_data["schema_version"], "1.0")
            self.assertTrue(batch_data["batch_id"].startswith("batch_"))
            self.assertEqual(batch_data["source_query"], "https://www.threads.net/@user/post/C_abc123")
            self.assertFalse(batch_data["verified"])
            self.assertEqual(len(batch_data["items"]), 1)

            item = batch_data["items"][0]
            self.assertEqual(item["comment_id"], "th_981273")
            self.assertEqual(item["post_id"], "C_abc123")
            self.assertIn("[URL]", item["text_anonymized"])
            self.assertIn("@user_anon", item["text_anonymized"])
            self.assertFalse(item["verified"])
            self.assertFalse(item["discarded"])
            self.assertIsNone(item["reviewed_at"])

    def test_create_staging_batch_filters_russian_comments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_items = [
                {
                    "comment_id": "c_kk",
                    "post_id": "p1",
                    "text": "Каспиден жаңа тауар алдым, өте жақсы екен.",
                },
                {
                    "comment_id": "c_ru",
                    "post_id": "p1",
                    "text": "Здравствуйте, подскажите пожалуйста, сколько стоит доставка в Алматы?",
                },
            ]
            batch_path, batch_data = create_staging_batch(
                source_query="https://www.threads.net/@user/post/p1",
                raw_items=raw_items,
                output_dir=tmpdir,
                filter_non_kazakh=True,
            )

            items = batch_data["items"]
            self.assertEqual(len(items), 2)

            # Kazakh item is kept active
            self.assertEqual(items[0]["lang"], "kk")
            self.assertFalse(items[0]["discarded"])

            # Russian item is automatically marked discarded
            self.assertEqual(items[1]["lang"], "ru")
            self.assertTrue(items[1]["discarded"])
            self.assertIn("auto_filtered_ru", items[1]["discard_reason"])

    @patch("httpx.Client.get")
    def test_harvest_threads_url_mocked_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_THREADS_HTML_WITH_JSON_LD
        mock_get.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            result = harvest_threads_url(
                "https://www.threads.net/@user/post/C_test123",
                output_dir=tmpdir,
            )
            self.assertIsInstance(result, ThreadsHarvestResult)
            self.assertTrue(result.success)
            self.assertGreaterEqual(result.comments_count, 2)
            self.assertTrue(os.path.isfile(result.batch_file_path))

    @patch("httpx.Client.get")
    def test_harvest_threads_url_mocked_failure(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        mock_get.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            result = harvest_threads_url(
                "https://www.threads.net/@user/post/C_invalid404",
                output_dir=tmpdir,
            )
            self.assertIsInstance(result, ThreadsHarvestResult)
            self.assertFalse(result.success)
            self.assertEqual(result.comments_count, 0)
            self.assertIn("404", result.error_message)


if __name__ == "__main__":
    unittest.main()
