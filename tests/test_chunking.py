import io
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import utils
import engine
import dataserver_cli


class SplitIntoSentencesTests(unittest.TestCase):
    def test_bullets_do_not_absorb_following_paragraphs_or_headings(self):
        text = (
            "Overview sentence.\n\n"
            "- First bullet sentence.\n"
            "- Final bullet sentence.\n\n"
            "## Later section\n\n"
            "Following paragraph sentence. Another sentence."
        )

        sentences = utils.split_into_sentences(text)

        self.assertEqual(
            sentences,
            [
                "Overview sentence.",
                "- First bullet sentence.",
                "- Final bullet sentence.",
                "## Later section\n\nFollowing paragraph sentence.",
                "Another sentence.",
            ],
        )
        self.assertNotIn("Following paragraph", sentences[2])
        self.assertNotIn("## Later section", sentences[2])

    def test_chunking_keeps_following_prose_separate_from_last_bullet(self):
        text = "- Bullet sentence.\n\n" + "Later prose sentence. " * 20

        chunks = utils.chunk_text_by_sentences(text, chunk_size=40)

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 40 for chunk in chunks))
        self.assertTrue(any("Later prose" in chunk for chunk in chunks))

    def test_prose_between_bullet_items_is_split_outside_bullet_segment(self):
        text = (
            "- First bullet sentence.\n\n"
            "Intervening prose sentence. Another prose sentence.\n"
            "## Later section\n\n"
            "- Second bullet sentence."
        )

        sentences = utils.split_into_sentences(text)

        self.assertEqual(
            sentences,
            [
                "- First bullet sentence.",
                "Intervening prose sentence.",
                "Another prose sentence.",
                "## Later section",
                "- Second bullet sentence.",
            ],
        )


class SynthesizeTokenLimitTests(unittest.TestCase):
    def test_non_cli_call_keeps_none_result_when_inference_fails(self):
        class Session:
            def get_providers(self): ...

            def run(self, output_names, inputs): ...

        session = MagicMock(spec=Session)
        session.get_providers.return_value = ["CPUExecutionProvider"]
        session.run.side_effect = RuntimeError("inference failed")

        with (
            patch.object(engine, "MODEL_LOADED", True),
            patch.object(engine, "onnx_session", session),
            patch.object(engine, "voices_data", {"voice": np.zeros((1, 4))}),
            patch.object(engine, "phonemizer_backend") as backend,
            patch.object(engine, "text_cleaner", lambda text: list(range(len(text)))),
            patch.object(engine, "get_all_accepted_voices", return_value=["voice"]),
        ):
            backend.phonemize.return_value = ["hello"]

            self.assertEqual(engine.synthesize("hello", "voice"), (None, None))
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                engine.synthesize("hello", "voice", raise_on_error=True)

    def test_oversized_phonemes_are_split_and_audio_is_joined_in_order(self):
        class Session:
            def get_providers(self): ...

            def run(self, output_names, inputs): ...

        session = MagicMock(spec=Session)
        submitted_lengths = []

        def infer(_output_names, inputs):
            submitted_lengths.append(inputs["input_ids"].shape[1])
            return [np.full((1, 2), len(submitted_lengths), dtype=np.float32)]

        session.get_providers.return_value = ["CPUExecutionProvider"]
        session.run.side_effect = infer

        with (
            patch.object(engine, "MODEL_LOADED", True),
            patch.object(engine, "onnx_session", session),
            patch.object(engine, "voices_data", {"voice": np.zeros((1, 4))}),
            patch.object(engine, "_voice_aliases", {"Bruno": "voice"}),
            patch.object(engine, "phonemizer_backend") as backend,
            patch.object(engine, "text_cleaner", lambda text: list(range(len(text)))),
            patch.object(engine, "get_all_accepted_voices", return_value=["voice"]),
        ):
            backend.phonemize.side_effect = lambda texts: texts

            audio, sample_rate = engine.synthesize("word " * 400, "voice")

        self.assertTrue(submitted_lengths)
        self.assertTrue(all(length <= 500 for length in submitted_lengths))
        self.assertGreater(len(submitted_lengths), 1)
        self.assertEqual(sample_rate, 24000)
        np.testing.assert_array_equal(
            audio,
            np.repeat(np.arange(1, len(submitted_lengths) + 1), 2).reshape(1, -1),
        )


class DataserverCliTests(unittest.TestCase):
    def test_synthesis_failure_includes_chunk_and_original_cause(self):
        with (
            patch.object(dataserver_cli.engine, "get_available_voices", return_value=["Bruno"]),
            patch.object(dataserver_cli.utils, "chunk_text_by_sentences", return_value=["failing text"]),
            patch.object(
                dataserver_cli.engine,
                "synthesize",
                side_effect=RuntimeError("invalid expand shape"),
            ) as synthesize,
        ):
            with self.assertRaisesRegex(RuntimeError, "failing text.*invalid expand shape"):
                dataserver_cli.synth("failing text", "output")

        synthesize.assert_called_once_with(
            "failing text", "Bruno", dataserver_cli.DEFAULT_SPEED, raise_on_error=True
        )

    def test_serve_replies_with_success_and_failure(self):
        output = io.StringIO()
        requests = '{"text":"hello","out":"output"}\n{"text":"bad","out":"output"}\n'

        with (
            patch.object(dataserver_cli.engine, "load_model", return_value=True),
            patch.object(dataserver_cli, "synth", side_effect=[None, RuntimeError("model error")]),
            patch("sys.stdin", io.StringIO(requests)),
            patch("sys.stdout", output),
        ):
            self.assertEqual(dataserver_cli.serve(), 0)

        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(replies, [{"ready": True}, {"ok": True}, {"ok": False, "error": "model error"}])

    def test_two_thousand_character_request_produces_bounded_audio(self):
        class Session:
            def get_providers(self): ...

            def run(self, output_names, inputs): ...

        session = MagicMock(spec=Session)
        submitted_lengths = []

        def infer(_output_names, inputs):
            submitted_lengths.append(inputs["input_ids"].shape[1])
            return [np.ones((1, 4), dtype=np.float32)]

        session.get_providers.return_value = ["CPUExecutionProvider"]
        session.run.side_effect = infer
        written = {}

        with (
            patch.object(dataserver_cli.engine, "get_available_voices", return_value=["Bruno"]),
            patch.object(engine, "MODEL_LOADED", True),
            patch.object(engine, "onnx_session", session),
            patch.object(engine, "voices_data", {"voice": np.zeros((1, 4))}),
            patch.object(engine, "_voice_aliases", {"Bruno": "voice"}),
            patch.object(engine, "phonemizer_backend") as backend,
            patch.object(engine, "text_cleaner", lambda text: list(range(len(text)))),
            patch.object(engine, "get_all_accepted_voices", return_value=["Bruno"]),
            patch.object(dataserver_cli.os, "makedirs"),
            patch.object(dataserver_cli.sf, "write", side_effect=lambda path, audio, *a, **kw: written.update(path=path, audio=audio)),
        ):
            backend.phonemize.side_effect = lambda texts: texts
            dataserver_cli.synth("word " * 400, "output")

        self.assertTrue(submitted_lengths)
        self.assertGreater(len(submitted_lengths), 1)
        self.assertTrue(all(length <= engine.MODEL_TOKEN_LIMIT for length in submitted_lengths))
        self.assertEqual(written["path"].replace("\\", "/"), "output/speech.mp3")
        self.assertGreater(written["audio"].size, 0)


if __name__ == "__main__":
    unittest.main()
