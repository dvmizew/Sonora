from unittest.mock import MagicMock, patch

from sonora.core.rate_limit import RateLimiter
from tests.support import IsolatedTestCase


class TestNetworkRetries(IsolatedTestCase):
    def test_retry_network_call_success_on_first_try(self) -> None:
        from sonora.core.utils import retry_network_call

        mock_func = MagicMock(return_value="ok")
        result = retry_network_call(mock_func)
        self.assertEqual(result, "ok")
        self.assertEqual(mock_func.call_count, 1)

    def test_retry_network_call_retries_and_succeeds(self) -> None:
        from sonora.core.utils import retry_network_call

        limiter = RateLimiter(interval_seconds=0)
        mock_func = MagicMock(
            side_effect=[OSError("Connection lost"), OSError("Timeout"), "recovered"]
        )
        with patch("time.sleep") as mock_sleep:
            result = retry_network_call(
                mock_func,
                max_retries=3,
                initial_delay=0.01,
                rate_limiter=limiter,
            )
        self.assertEqual(result, "recovered")
        self.assertEqual(mock_func.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    def test_retry_network_call_exhausts_retries(self) -> None:
        from sonora.core.utils import retry_network_call

        mock_func = MagicMock(side_effect=OSError("Permanent failure"))
        with patch("time.sleep"), self.assertRaises(OSError):
            retry_network_call(mock_func, max_retries=3, initial_delay=0.01)
        self.assertEqual(mock_func.call_count, 3)

    def test_retry_network_call_skips_non_retryable(self) -> None:
        from sonora.core.utils import retry_network_call

        mock_func = MagicMock(side_effect=ValueError("Invalid parameter"))
        with patch("time.sleep"), self.assertRaises(ValueError):
            retry_network_call(
                mock_func,
                max_retries=3,
                transient_exceptions=(OSError,),
            )
        self.assertEqual(mock_func.call_count, 1)

    def test_retry_network_call_respects_retry_condition(self) -> None:
        from sonora.core.utils import retry_network_call

        mock_func = MagicMock(side_effect=OSError("Forbidden"))
        with patch("time.sleep"), self.assertRaises(OSError):
            retry_network_call(
                mock_func,
                max_retries=3,
                retry_condition=lambda err: "Forbidden" not in str(err),
            )
        self.assertEqual(mock_func.call_count, 1)

    def test_retry_transport_retries_on_429_and_503(self) -> None:
        import httpx

        from sonora.core.http import RetryTransport

        resp_429 = httpx.Response(
            429,
            headers={"Retry-After": "2"},
            request=httpx.Request("GET", "https://example.com"),
        )
        resp_503 = httpx.Response(
            503, request=httpx.Request("GET", "https://example.com")
        )
        resp_200 = httpx.Response(
            200,
            json={"status": "ok"},
            request=httpx.Request("GET", "https://example.com"),
        )

        transport = RetryTransport(max_retries=3, backoff_factor=0.01)

        with (
            patch.object(
                httpx.HTTPTransport,
                "handle_request",
                side_effect=[resp_429, resp_503, resp_200],
            ) as mock_handle,
            patch("time.sleep") as mock_sleep,
        ):
            response = transport.handle_request(
                httpx.Request("GET", "https://example.com")
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_handle.call_count, 3)
        mock_sleep.assert_any_call(2.0)

    def test_retry_transport_retries_on_transport_error(self) -> None:
        import httpx

        from sonora.core.http import RetryTransport

        resp_200 = httpx.Response(
            200,
            json={"status": "ok"},
            request=httpx.Request("GET", "https://example.com"),
        )
        transport = RetryTransport(max_retries=2, backoff_factor=0.01)

        with (
            patch.object(
                httpx.HTTPTransport,
                "handle_request",
                side_effect=[httpx.ConnectTimeout("Timed out"), resp_200],
            ) as mock_handle,
            patch("time.sleep"),
        ):
            response = transport.handle_request(
                httpx.Request("GET", "https://example.com")
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_handle.call_count, 2)
