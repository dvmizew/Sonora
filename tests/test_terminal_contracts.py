import threading
import time
from collections import deque
from unittest.mock import Mock, patch

from rich.progress import Progress, Task, TaskID

from sonora.core import logger
from tests.support import IsolatedTestCase


class TerminalScript:
    def __init__(
        self,
        keys: tuple[str | None, ...],
        stop_event: threading.Event,
        *,
        stop_on_ready: bool = False,
        drain_key: str | None = None,
    ) -> None:
        self.keys = deque(keys)
        self.stop_event = stop_event
        self.stop_on_ready = stop_on_ready
        self.drain_key = drain_key
        self.current_key = ""

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 17

    def read(self, size: int) -> str:
        return self.current_key

    def select(
        self,
        readers: list[object],
        writers: list[object],
        errors: list[object],
        timeout: float,
    ) -> tuple[list[object], list[object], list[object]]:
        if timeout == 0.05:
            if self.drain_key is not None:
                self.current_key, self.drain_key = self.drain_key, None
                return [self], [], []
            return [], [], []
        if not self.keys:
            self.stop_event.set()
            return [], [], []
        key = self.keys.popleft()
        if key is None:
            return [], [], []
        self.current_key = key
        if self.stop_on_ready:
            self.stop_event.set()
        return [self], [], []


class TestTerminalContracts(IsolatedTestCase):
    def test_debug_messages_redact_credentials_and_escape_markup(self) -> None:
        instance = logger.SonoraLogger()
        instance.verbose = True
        with patch.object(logger.CONSOLE, "print") as output:
            instance.debug(
                "[red] token=secret&api_key=private client_key=hidden user_token=another"
            )
        message = output.call_args.args[0]
        self.assertNotIn("secret", message)
        self.assertNotIn("private", message)
        self.assertNotIn("hidden", message)
        self.assertNotIn("another", message)
        self.assertIn("<redacted>", message)
        self.assertIn("\\[red]", message)

    def test_error_and_buffer_lifecycle_preserve_message_order(self) -> None:
        instance = logger.SonoraLogger()
        with patch.object(logger.CONSOLE, "print") as output:
            instance.stop_buffering()
            output.assert_not_called()
            instance.start_buffering()
            instance.error("broken track")
            instance.info("next recording")
            output.assert_not_called()
            instance.stop_buffering()
        messages = [call.args[0] for call in output.call_args_list]
        self.assertIn("ERROR:", messages[0])
        self.assertIn("broken track", messages[0])
        self.assertIn("next recording", messages[1])

    def test_worker_pause_waits_until_resume(self) -> None:
        with patch.object(
            logger._PAUSE_EVENT, "wait", side_effect=[False, False, True]
        ) as wait:
            logger.wait_if_paused(0.01)
        self.assertEqual(wait.call_count, 3)
        self.assertEqual(wait.call_args.kwargs, {"timeout": 0.01})

    def test_default_pause_resume_corrects_task_clock(self) -> None:
        stop = threading.Event()
        terminal = TerminalScript((None, "x", "p", "P"), stop, drain_key="held key")
        task = Task(TaskID(3), "Tagging", 2, 0, time.monotonic)
        task.start_time = 20
        task.stop_time = 21
        progress = Mock(spec=Progress(auto_refresh=False))
        progress.tasks = [task]
        progress.get_time.side_effect = [100, 112]
        with (
            patch("sys.stdin", terminal),
            patch("threading.Event", return_value=stop),
            patch("threading.Thread") as thread,
            patch("select.select", side_effect=terminal.select),
            patch("termios.tcgetattr", return_value=[1, 2, 3]) as get_attributes,
            patch("termios.tcsetattr") as restore,
            patch("tty.setcbreak") as cbreak,
            patch.object(logger.LOG, "warning"),
            patch.object(logger.LOG, "info"),
        ):
            with logger.interactive_pause_listener(progress, TaskID(3)):
                thread.call_args.kwargs["target"]()
            get_attributes.assert_called_once_with(17)
            cbreak.assert_called_once_with(17)
            restore.assert_called_once()
            thread.return_value.join.assert_called_once_with(timeout=0.5)
        self.assertEqual(task.start_time, 32)
        self.assertIsNone(task.stop_time)
        progress.stop_task.assert_called_once_with(TaskID(3))
        self.assertEqual(progress.update.call_args.kwargs["description"], "Tagging")
        self.assertTrue(logger._PAUSE_EVENT.is_set())

    def test_missing_task_has_fallback_description_and_no_clock_update(self) -> None:
        task_variants: tuple[list[Task], ...] = (
            [],
            [Task(TaskID(3), "Tagging", 2, 0, time.monotonic)],
        )
        for tasks in task_variants:
            stop = threading.Event()
            terminal = TerminalScript((" ", " "), stop)
            progress = Mock(spec=Progress(auto_refresh=False))
            progress.tasks = tasks
            progress.get_time.side_effect = [5, 9]
            with (
                self.subTest(tasks=tasks),
                patch("sys.stdin", terminal),
                patch("threading.Event", return_value=stop),
                patch("threading.Thread") as thread,
                patch("select.select", side_effect=terminal.select),
                patch("termios.tcgetattr", return_value=[]),
                patch("termios.tcsetattr"),
                patch("tty.setcbreak"),
                patch.object(logger.LOG, "warning"),
                patch.object(logger.LOG, "info"),
                logger.interactive_pause_listener(progress, TaskID(3)),
            ):
                thread.call_args.kwargs["target"]()
            expected_description = "Tagging" if tasks else "[cyan]Processing..."
            self.assertEqual(
                progress.update.call_args.kwargs["description"], expected_description
            )
            if tasks:
                self.assertIsNone(tasks[0].start_time)

    def test_callbacks_handle_expected_errors_and_resume_without_default_pause(
        self,
    ) -> None:
        for pause_error, resume_error in (
            (None, None),
            (OSError("pause"), ValueError("resume")),
            (RuntimeError("pause"), RuntimeError("resume")),
        ):
            stop = threading.Event()
            terminal = TerminalScript(("p", "p"), stop)
            pause = Mock(side_effect=pause_error)
            resume = Mock(side_effect=resume_error)
            with (
                self.subTest(pause_error=pause_error),
                patch("sys.stdin", terminal),
                patch("threading.Event", return_value=stop),
                patch("threading.Thread") as thread,
                patch("select.select", side_effect=terminal.select),
                patch("termios.tcgetattr", return_value=[]),
                patch("termios.tcsetattr"),
                patch("tty.setcbreak"),
                patch.object(logger.LOG, "warning"),
                patch.object(logger.LOG, "info"),
                patch.object(logger.LOG, "debug") as debug,
            ):
                with logger.interactive_pause_listener(
                    on_pause=pause, on_resume=resume
                ):
                    thread.call_args.kwargs["target"]()
                pause.assert_called_once()
                resume.assert_called_once()
                self.assertEqual(debug.call_count, 0 if pause_error is None else 2)
            self.assertTrue(logger._PAUSE_EVENT.is_set())

    def test_custom_pause_with_default_resume_does_not_shift_task_time(self) -> None:
        stop = threading.Event()
        terminal = TerminalScript(("p", "p"), stop)
        progress = Mock(spec=Progress(auto_refresh=False))
        progress.tasks = []
        pause = Mock()
        with (
            patch("sys.stdin", terminal),
            patch("threading.Event", return_value=stop),
            patch("threading.Thread") as thread,
            patch("select.select", side_effect=terminal.select),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr"),
            patch("tty.setcbreak"),
            patch.object(logger.LOG, "warning"),
            patch.object(logger.LOG, "info"),
            logger.interactive_pause_listener(progress, TaskID(3), on_pause=pause),
        ):
            thread.call_args.kwargs["target"]()
        pause.assert_called_once()
        progress.get_time.assert_not_called()
        self.assertEqual(
            progress.update.call_args.kwargs["description"], "[cyan]Processing..."
        )

    def test_listener_ignores_key_ready_after_shutdown(self) -> None:
        stop = threading.Event()
        terminal = TerminalScript(("p",), stop, stop_on_ready=True)
        pause = Mock()
        with (
            patch("sys.stdin", terminal),
            patch("threading.Event", return_value=stop),
            patch("threading.Thread") as thread,
            patch("select.select", side_effect=terminal.select),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr", side_effect=OSError("terminal closed")),
            patch("tty.setcbreak"),
            logger.interactive_pause_listener(on_pause=pause),
        ):
            thread.call_args.kwargs["target"]()
        pause.assert_not_called()
        self.assertTrue(logger._PAUSE_EVENT.is_set())

    def test_terminal_setup_failure_preserves_worker_resume(self) -> None:
        stop = threading.Event()
        terminal = TerminalScript((), stop)
        with (
            patch("sys.stdin", terminal),
            patch("threading.Event", return_value=stop),
            patch("threading.Thread") as thread,
            patch("termios.tcgetattr", side_effect=OSError("no terminal")),
            logger.interactive_pause_listener(),
        ):
            thread.call_args.kwargs["target"]()
        self.assertTrue(stop.is_set())
        self.assertTrue(logger._PAUSE_EVENT.is_set())

    def test_default_callbacks_without_progress_only_toggle_pause(self) -> None:
        stop = threading.Event()
        terminal = TerminalScript(("p", "p"), stop)
        with (
            patch("sys.stdin", terminal),
            patch("threading.Event", return_value=stop),
            patch("threading.Thread") as thread,
            patch("select.select", side_effect=terminal.select),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr"),
            patch("tty.setcbreak"),
            patch.object(logger.LOG, "warning") as paused,
            patch.object(logger.LOG, "info") as resumed,
            logger.interactive_pause_listener(),
        ):
            thread.call_args.kwargs["target"]()
        paused.assert_called_once()
        resumed.assert_called_once()
        self.assertTrue(logger._PAUSE_EVENT.is_set())
