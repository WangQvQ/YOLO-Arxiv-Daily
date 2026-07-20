import unittest
from unittest.mock import Mock, patch

import arxiv

from daily_arxiv import _fetch_results


class FetchResultsTest(unittest.TestCase):
    def test_returns_results_without_retry(self):
        client = Mock()
        client.results.return_value = iter(["paper"])

        self.assertEqual(_fetch_results(client, Mock()), ["paper"])
        client.results.assert_called_once()

    @patch("daily_arxiv.time.sleep")
    def test_retries_429_with_exponential_backoff(self, sleep):
        client = Mock()
        client.results.side_effect = [
            arxiv.HTTPError("https://example.test", 0, 429),
            arxiv.HTTPError("https://example.test", 1, 429),
            iter(["paper"]),
        ]

        results = _fetch_results(client, Mock(), max_retries=3, initial_delay=1)

        self.assertEqual(results, ["paper"])
        self.assertEqual(sleep.call_args_list, [unittest.mock.call(1), unittest.mock.call(2)])

    @patch("daily_arxiv.time.sleep")
    def test_reraises_non_429_without_retry(self, sleep):
        client = Mock()
        error = arxiv.HTTPError("https://example.test", 0, 500)
        client.results.side_effect = error

        with self.assertRaises(arxiv.HTTPError) as raised:
            _fetch_results(client, Mock())

        self.assertIs(raised.exception, error)
        client.results.assert_called_once()
        sleep.assert_not_called()

    @patch("daily_arxiv.time.sleep")
    def test_reraises_429_after_retry_limit(self, sleep):
        client = Mock()
        error = arxiv.HTTPError("https://example.test", 2, 429)
        client.results.side_effect = error

        with self.assertRaises(arxiv.HTTPError) as raised:
            _fetch_results(client, Mock(), max_retries=3, initial_delay=1)

        self.assertIs(raised.exception, error)
        self.assertEqual(client.results.call_count, 3)
        self.assertEqual(sleep.call_args_list, [unittest.mock.call(1), unittest.mock.call(2)])


if __name__ == "__main__":
    unittest.main()
