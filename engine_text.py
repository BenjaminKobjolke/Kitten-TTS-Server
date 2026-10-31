"""Text cleanup and tokenization for KittenTTS inference."""

import re


class TextCleaner:
    """Convert phoneme characters to model token indices."""

    def __init__(self):
        pad = "$"
        punctuation = ';:,.!?¡¿—…”«»"" '
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        letters_ipa = (
            "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻ"
            "ʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        )
        symbols = [pad] + list(punctuation) + list(letters) + list(letters_ipa)
        self.word_index_dictionary = {
            symbol: index for index, symbol in enumerate(symbols)
        }

    def __call__(self, text: str):
        return [
            self.word_index_dictionary[char]
            for char in text
            if char in self.word_index_dictionary
        ]


def basic_english_tokenize(text: str):
    """Split text on whitespace and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text)
