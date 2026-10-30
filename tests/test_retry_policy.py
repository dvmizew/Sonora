from unittest.mock import MagicMock, patch

from sonora.core.rate_limit import RateLimiter
from tests.support import IsolatedTestCase


class TestNetworkRetries(IsolatedTestCase):
    def test_lrclib_workers_send_requests_sequentially(self) -> None:
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor

        import httpx

        from sonora.core.http import _HOST_LIMITERS, RetryTransport

        active = 0
        maximum = 0
        lock = threading.Lock()
        request = httpx.Request("GET", "https://lrclib.net/api/get")

        def send(request: httpx.Request) -> httpx.Response:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return httpx.Response(200, request=request)

        with (
            RetryTransport() as transport,
            patch.object(_HOST_LIMITERS["lrclib.net"], "wait"),
            patch.object(httpx.HTTPTransport, "handle_request", side_effect=send),
            ThreadPoolExecutor(max_workers=4) as pool,
        ):
            responses = list(pool.map(transport.handle_request, [request] * 4))
        self.assertEqual([response.status_code for response in responses], [200] * 4)
        self.assertEqual(maximum, 1)
        self.assertGreaterEqual(_HOST_LIMITERS["lrclib.net"].interval, 1.0)

    def test_provider_cooldown_is_shared_and_cannot_be_shortened(self) -> None:
        from sonora.core.coordination import wait_for_provider

        with (
            patch(
                "sonora.core.coordination.get_cache_dir",
                return_value=self.test_directory,
            ),
            patch("sonora.core.coordination.time.time", return_value=100.0),
            patch("sonora.core.coordination.time.sleep") as sleep,
        ):
            wait_for_provider("lrclib.net", 0, defer_seconds=30)
            wait_for_provider("lrclib.net", 0, defer_seconds=1)
            sleep.assert_not_called()
            self.assertEqual(wait_for_provider("lrclib.net", 0.25), 30)
            sleep.assert_called_once_with(30)

    def test_final_rate_limit_response_sets_shared_cooldown(self) -> None:
        import httpx

        from sonora.core.http import _HOST_LIMITERS, RetryTransport

        request = httpx.Request("GET", "https://lrclib.net/api/get")
        response = httpx.Response(429, headers={"Retry-After": "30"}, request=request)
        transport = RetryTransport(max_retries=1)
        RateLimiter.set_disabled(False)
        try:
            with (
                patch.object(_HOST_LIMITERS["lrclib.net"], "wait"),
                patch.object(
                    httpx.HTTPTransport, "handle_request", return_value=response
                ),
                patch("sonora.core.http.wait_for_provider") as defer,
                patch("sonora.core.http.SystemRandom") as random,
            ):
                random.return_value.uniform.return_value = 3.0
                self.assertIs(transport.handle_request(request), response)
                defer.assert_called_once_with("lrclib.net", 0, defer_seconds=33)
        finally:
            transport.close()
            RateLimiter.set_disabled(True)

    def test_retry_network_call_success_on_first_try(self) -> None:
        from sonora.core.retry import retry_network_call

        mock_func = MagicMock(return_value="ok")
        result = retry_network_call(mock_func)
        self.assertEqual(result, "ok")
        self.assertEqual(mock_func.call_count, 1)

    def test_retry_network_call_retries_and_succeeds(self) -> None:
        from sonora.core.retry import retry_network_call

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
        from sonora.core.retry import retry_network_call

        mock_func = MagicMock(side_effect=OSError("Permanent failure"))
        with patch("time.sleep"), self.assertRaises(OSError):
            retry_network_call(mock_func, max_retries=3, initial_delay=0.01)
        self.assertEqual(mock_func.call_count, 3)

    def test_retry_network_call_skips_non_retryable(self) -> None:
        from sonora.core.retry import retry_network_call

        mock_func = MagicMock(side_effect=ValueError("Invalid parameter"))
        with patch("time.sleep"), self.assertRaises(ValueError):
            retry_network_call(
                mock_func,
                max_retries=3,
                transient_exceptions=(OSError,),
            )
        self.assertEqual(mock_func.call_count, 1)

    def test_retry_network_call_respects_retry_condition(self) -> None:
        from sonora.core.retry import retry_network_call

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
