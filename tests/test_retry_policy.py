from unittest.mock import MagicMock, patch

from sonora.core.rate_limit import RateLimiter
from tests.support import IsolatedTestCase


class TestNetworkRetries(IsolatedTestCase):
    def test_lrclib_workers_wait_for_each_response_before_next_request(self) -> None:
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
            try:
                time.sleep(0.01)
            finally:
                with lock:
                    active -= 1
            return httpx.Response(200, request=request)

        with (
            RetryTransport() as transport,
            patch.object(_HOST_LIMITERS["lrclib.net"], "wait") as pace_request,
            patch.object(httpx.HTTPTransport, "handle_request", side_effect=send),
            ThreadPoolExecutor(max_workers=4) as pool,
        ):
            responses = list(pool.map(transport.handle_request, [request] * 4))
        self.assertEqual([response.status_code for response in responses], [200] * 4)
        self.assertEqual(maximum, 1)
        self.assertEqual(pace_request.call_count, 4)
        self.assertEqual(_HOST_LIMITERS["lrclib.net"].interval, 0.3)

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

    def test_long_provider_cooldown_defers_workers_without_waiting(self) -> None:
        from sonora.core.coordination import (
            ProviderDeferredError,
            wait_for_provider,
        )

        with (
            patch(
                "sonora.core.coordination.get_cache_dir",
                return_value=self.test_directory,
            ),
            patch("sonora.core.coordination.time.time", return_value=100.0),
            patch("sonora.core.coordination.time.sleep") as sleep,
        ):
            wait_for_provider("lrclib.net", 0, defer_seconds=3600)
            with self.assertRaises(ProviderDeferredError):
                wait_for_provider("lrclib.net", 0.25)
        sleep.assert_not_called()

    def test_retry_transport_does_not_retry_a_deferred_provider(self) -> None:
        import httpx

        from sonora.core.coordination import ProviderDeferredError
        from sonora.core.http import _HOST_LIMITERS, RetryTransport

        with (
            patch.object(
                _HOST_LIMITERS["lrclib.net"],
                "wait",
                side_effect=ProviderDeferredError("provider cooling down"),
            ),
            patch.object(httpx.HTTPTransport, "handle_request") as send,
            RetryTransport(max_retries=3) as transport,
            self.assertRaises(ProviderDeferredError),
        ):
            transport.handle_request(httpx.Request("GET", "https://lrclib.net/api"))
        send.assert_not_called()

    def test_final_rate_limit_response_sets_shared_cooldown(self) -> None:
        import httpx

        from sonora.core.http import _HOST_LIMITERS, RetryTransport

        request = httpx.Request("GET", "https://lrclib.net/api/get")
        response = httpx.Response(429, headers={"Retry-After": "30"}, request=request)
        with (
            RetryTransport(max_retries=1) as transport,
            patch.object(RateLimiter, "_disabled", False),
            patch.object(_HOST_LIMITERS["lrclib.net"], "wait"),
            patch.object(httpx.HTTPTransport, "handle_request", return_value=response),
            patch("sonora.core.http.wait_for_provider") as defer,
            patch("sonora.core.http.SystemRandom") as random,
        ):
            random.return_value.uniform.return_value = 3.0
            self.assertIs(transport.handle_request(request), response)
            defer.assert_called_once_with("lrclib.net", 0, defer_seconds=33)

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

    def test_retry_network_call_skips_dns_lookup_failures(self) -> None:
        import socket

        from musicbrainzngs import NetworkError

        from sonora.core.retry import retry_network_call

        for code in (socket.EAI_AGAIN, socket.EAI_NONAME):
            with self.subTest(code=code):
                failure = NetworkError(
                    "MusicBrainz network error",
                    cause=socket.gaierror(code, "DNS lookup failed"),
                )
                mock_func = MagicMock(side_effect=failure)
                with (
                    patch("time.sleep") as sleep,
                    patch(
                        "sonora.core.retry._DNS_UNAVAILABLE_UNTIL",
                        0.0,
                        create=True,
                    ),
                    self.assertRaises(NetworkError),
                ):
                    retry_network_call(mock_func, transient_exceptions=(NetworkError,))
                mock_func.assert_called_once()
                sleep.assert_not_called()

    def test_dns_failure_suspends_repeated_network_calls_briefly(self) -> None:
        import socket

        from sonora.core.retry import retry_network_call

        first = MagicMock(side_effect=socket.gaierror(socket.EAI_AGAIN, "DNS failed"))
        later = MagicMock(return_value="unexpected")
        with self.assertRaises(socket.gaierror):
            retry_network_call(first)
        with self.assertRaises(socket.gaierror):
            retry_network_call(later)
        first.assert_called_once()
        later.assert_not_called()

    def test_retry_network_call_rechecks_dns_after_rate_limit_wait(self) -> None:
        import socket

        from sonora.core.retry import _suspend_dns_requests, retry_network_call

        limiter = RateLimiter(interval_seconds=0)
        request = MagicMock(return_value="unexpected")
        with (
            patch("sonora.core.retry._DNS_UNAVAILABLE_UNTIL", 0.0),
            patch.object(limiter, "wait", side_effect=_suspend_dns_requests),
            self.assertRaises(socket.gaierror),
        ):
            retry_network_call(request, rate_limiter=limiter)
        request.assert_not_called()

    def test_permanent_dns_failure_does_not_suspend_other_lookups(self) -> None:
        import socket

        from sonora.core.retry import retry_network_call

        failure = MagicMock(
            side_effect=socket.gaierror(socket.EAI_NONAME, "Unknown host")
        )
        available = MagicMock(return_value="available")
        with self.assertRaises(socket.gaierror):
            retry_network_call(failure)
        self.assertEqual(retry_network_call(available), "available")
        failure.assert_called_once()
        available.assert_called_once()

    def test_dns_cooldown_expires(self) -> None:
        from sonora.core.retry import _dns_requests_suspended, _suspend_dns_requests

        with (
            patch("sonora.core.retry.time.monotonic", side_effect=[5.0, 34.9, 35.0]),
        ):
            _suspend_dns_requests()
            self.assertTrue(_dns_requests_suspended())
            self.assertFalse(_dns_requests_suspended())

    def test_http_transport_skips_requests_during_dns_cooldown(self) -> None:
        import socket

        import httpx

        from sonora.core.http import RetryTransport

        request = httpx.Request("GET", "https://example.com")
        failure = httpx.ConnectError("DNS failed", request=request)
        failure.__cause__ = socket.gaierror(socket.EAI_AGAIN, "DNS failed")
        with (
            RetryTransport() as transport,
            patch.object(
                httpx.HTTPTransport, "handle_request", side_effect=failure
            ) as send,
        ):
            with self.assertRaises(httpx.ConnectError):
                transport.handle_request(request)
            with self.assertRaises(httpx.ConnectError):
                transport.handle_request(request)
        self.assertEqual(send.call_count, 1)

    def test_http_transport_rechecks_dns_after_rate_limit_wait(self) -> None:
        import httpx

        from sonora.core.http import _HOST_LIMITERS, RetryTransport
        from sonora.core.retry import _suspend_dns_requests

        request = httpx.Request("GET", "https://lrclib.net/api/get")
        with (
            RetryTransport() as transport,
            patch("sonora.core.retry._DNS_UNAVAILABLE_UNTIL", 0.0),
            patch.object(
                _HOST_LIMITERS["lrclib.net"],
                "wait",
                side_effect=lambda *_args: _suspend_dns_requests(),
            ),
            patch.object(httpx.HTTPTransport, "handle_request") as send,
            self.assertRaises(httpx.ConnectError),
        ):
            transport.handle_request(request)
        send.assert_not_called()

    def test_http_transport_does_not_retry_after_another_request_suspends_dns(
        self,
    ) -> None:
        import httpx

        from sonora.core.http import RetryTransport
        from sonora.core.retry import _suspend_dns_requests

        request = httpx.Request("GET", "https://example.com")

        def fail_during_dns_outage(request: httpx.Request) -> httpx.Response:
            _suspend_dns_requests()
            raise httpx.ConnectTimeout("resolver unavailable", request=request)

        with (
            RetryTransport(max_retries=2) as transport,
            patch.object(
                httpx.HTTPTransport,
                "handle_request",
                side_effect=fail_during_dns_outage,
            ) as send,
            patch("sonora.core.http.time.sleep") as sleep,
            self.assertRaises(httpx.ConnectTimeout),
        ):
            transport.handle_request(request)

        send.assert_called_once_with(request)
        sleep.assert_not_called()

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

        with (
            RetryTransport(max_retries=3, backoff_factor=0.01) as transport,
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

    def test_lrclib_503_without_retry_after_uses_one_second_fallback(self) -> None:
        import httpx

        from sonora.core.http import _HOST_LIMITERS, RetryTransport

        request = httpx.Request("GET", "https://lrclib.net/api/get")
        responses = [
            httpx.Response(503, request=request),
            httpx.Response(200, request=request),
        ]
        with (
            RetryTransport(max_retries=2) as transport,
            patch.object(_HOST_LIMITERS["lrclib.net"], "wait"),
            patch("sonora.core.http.wait_for_provider"),
            patch.object(httpx.HTTPTransport, "handle_request", side_effect=responses),
            patch("sonora.core.http.SystemRandom") as random,
            patch("time.sleep") as sleep,
        ):
            random.return_value.uniform.return_value = 0
            response = transport.handle_request(request)

        self.assertEqual(response.status_code, 200)
        sleep.assert_called_once_with(1.0)

    def test_lrclib_stops_after_one_retry_for_repeated_server_errors(self) -> None:
        import httpx

        from sonora.core.http import _HOST_LIMITERS, RetryTransport

        request = httpx.Request("GET", "https://lrclib.net/api/get")
        responses = [
            httpx.Response(502, request=request),
            httpx.Response(502, request=request),
            httpx.Response(200, request=request),
        ]
        with (
            RetryTransport() as transport,
            patch.object(_HOST_LIMITERS["lrclib.net"], "wait"),
            patch("sonora.core.http.wait_for_provider"),
            patch.object(
                httpx.HTTPTransport, "handle_request", side_effect=responses
            ) as request_handler,
            patch("sonora.core.http.SystemRandom") as random,
            patch("time.sleep"),
        ):
            random.return_value.uniform.return_value = 0
            response = transport.handle_request(request)

        self.assertEqual(response.status_code, 502)
        self.assertEqual(request_handler.call_count, 2)

    def test_retry_transport_retries_on_transport_error(self) -> None:
        import httpx

        from sonora.core.http import RetryTransport

        resp_200 = httpx.Response(
            200,
            json={"status": "ok"},
            request=httpx.Request("GET", "https://example.com"),
        )
        with (
            RetryTransport(max_retries=2, backoff_factor=0.01) as transport,
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
