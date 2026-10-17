from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import httpx

from sonora.core.constants import RATE_LIMIT_DISCOGS_UNAUTHENTICATED
from sonora.core.http import RetryTransport, _retry_delay
from tests.support import IsolatedTestCase


class TestRetryTransportContracts(IsolatedTestCase):
    def test_invalid_attempt_budget_is_rejected(self) -> None:
        for attempts in (0, -1):
            with self.subTest(attempts=attempts), self.assertRaises(ValueError):
                RetryTransport(max_retries=attempts)

    def test_discogs_rate_limit_depends_on_authentication(self) -> None:
        limiter = Mock()
        for headers, interval in (
            ({}, RATE_LIMIT_DISCOGS_UNAUTHENTICATED),
            ({"Authorization": "Discogs token=fixture"}, None),
        ):
            limiter.reset_mock()
            with (
                self.subTest(headers=headers),
                patch("sonora.core.http._HOST_LIMITERS", {"api.discogs.com": limiter}),
                patch(
                    "httpx.HTTPTransport.handle_request",
                    return_value=httpx.Response(200),
                ),
                RetryTransport() as transport,
            ):
                response = transport.handle_request(
                    httpx.Request(
                        "GET", "https://api.discogs.com/test", headers=headers
                    )
                )
            self.assertEqual(response.status_code, 200)
            limiter.wait.assert_called_once_with(interval)

    def test_long_retry_after_returns_response_without_sleeping(self) -> None:
        response = httpx.Response(429, headers={"Retry-After": "61"})
        with (
            patch("httpx.HTTPTransport.handle_request", return_value=response) as send,
            patch("sonora.core.http.time.sleep") as sleep,
            RetryTransport() as transport,
        ):
            self.assertIs(
                transport.handle_request(httpx.Request("GET", "https://example.com")),
                response,
            )
        send.assert_called_once()
        sleep.assert_not_called()

    def test_exhausted_transport_failure_preserves_original_exception(self) -> None:
        error = httpx.ReadTimeout("socket timed out")
        with (
            patch("httpx.HTTPTransport.handle_request", side_effect=error) as send,
            patch("sonora.core.http.time.sleep") as sleep,
            RetryTransport(max_retries=2, backoff_factor=0.25) as transport,
            self.assertRaises(httpx.ReadTimeout) as raised,
        ):
            transport.handle_request(httpx.Request("GET", "https://example.com"))
        self.assertIs(raised.exception, error)
        self.assertEqual(send.call_count, 2)
        sleep.assert_called_once_with(0.25)

    def test_retry_dates_support_naive_utc_and_timezone_aware_deadlines(self) -> None:
        instant = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
        for deadline in (
            instant + timedelta(seconds=7),
            (instant + timedelta(seconds=7)).replace(tzinfo=None),
        ):
            with (
                self.subTest(deadline=deadline),
                patch("sonora.core.http.parsedate_to_datetime", return_value=deadline),
                patch("sonora.core.http.datetime", wraps=datetime) as clock,
            ):
                clock.now.return_value = instant
                self.assertEqual(_retry_delay("HTTP-date", 1.5), 7)
