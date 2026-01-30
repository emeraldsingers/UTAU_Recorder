from __future__ import annotations

from typing import Dict


_BASIC: Dict[str, str] = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "を": "o", "ん": "n",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo", "ゎ": "wa",
}

_DIGRAPHS: Dict[str, str] = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ぎぇ": "gye", "きぇ": "kye",
    "しぇ": "she", "ちぇ": "che", "じぇ": "je",
    "てぃ": "ti", "でぃ": "di",
    "てゅ": "tyu", "でゅ": "dyu",
    "とぅ": "tu", "どぅ": "du",
    "つぁ": "tsa", "つぃ": "tsi", "つぇ": "tse", "つぉ": "tso",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo",
    "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "くぁ": "kwa", "くぃ": "kwi", "くぇ": "kwe", "くぉ": "kwo",
    "ぐぁ": "gwa", "ぐぃ": "gwi", "ぐぇ": "gwe", "ぐぉ": "gwo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "みぇ": "mye", "りぇ": "rye", "にぇ": "nye", "ひぇ": "hye",
    "びぇ": "bye", "ぴぇ": "pye",
    "いぇ": "ye",
}


def needs_romaji(text: str) -> bool:
    return any("\u3040" <= ch <= "\u30ff" for ch in text)


def kana_to_romaji(text: str) -> str:
    return "".join(kana_to_romaji_tokens(text))


def kana_to_romaji_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "っ":
            if i + 1 < len(text):
                nxt = _peek_romaji(text, i + 1)
                if nxt:
                    tokens.append(nxt[0])
            i += 1
            continue
        if ch == "ん":
            if i + 1 < len(text):
                nxt = _peek_romaji(text, i + 1)
                if nxt and nxt[0] in ("a", "i", "u", "e", "o", "y"):
                    tokens.append("n'")
                else:
                    tokens.append("n")
            else:
                tokens.append("n")
            i += 1
            continue
        if i + 1 < len(text):
            pair = text[i : i + 2]
            if pair in _DIGRAPHS:
                tokens.append(_DIGRAPHS[pair])
                i += 2
                continue
        if ch in _BASIC:
            tokens.append(_BASIC[ch])
        else:
            tokens.append(ch)
        i += 1
    return tokens


def _peek_romaji(text: str, idx: int) -> str:
    if idx + 1 < len(text):
        pair = text[idx : idx + 2]
        if pair in _DIGRAPHS:
            return _DIGRAPHS[pair]
    ch = text[idx]
    return _BASIC.get(ch, "")
