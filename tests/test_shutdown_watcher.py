import tempfile
import unittest
from pathlib import Path

from shutdown_watcher import check_shutdown_flag


class CheckShutdownFlagTests(unittest.TestCase):
    def test_returns_false_when_flag_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(check_shutdown_flag(Path(tmp) / "_shutdown"))

    def test_returns_true_and_consumes_flag_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            flag = Path(tmp) / "_shutdown"
            flag.touch()

            self.assertTrue(check_shutdown_flag(flag))
            self.assertFalse(flag.exists())


if __name__ == "__main__":
    unittest.main()
