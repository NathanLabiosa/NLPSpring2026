import random
import string
import re

class PerturbationEngine:
    def __init__(self):
        # homophones map for semantic perturbations
        self.homophones_map = {
            "their": ["there", "they're"], "there": ["their", "they're"], "they're": ["their", "there"],
            "your": ["you're"], "you're": ["your"], "its": ["it's"], "it's": ["its"],
            "to": ["too", "two"], "too": ["to", "two"], "two": ["to", "too"],
            "then": ["than"], "than": ["then"], "weather": ["whether"], "whether": ["weather"],
            "write": ["right"], "right": ["write"], "read": ["red"], "red": ["read"],
            "for": ["four"], "four": ["for"], "sun": ["son"], "son": ["sun"]
        }

        # OCR visual lookalikes
        self.ocr_map = {
            'l': '1', '1': 'l', 'I': '1', 'O': '0', '0': 'O', 'S': '5', '5': 'S',
            'B': '8', '8': 'B', 'Z': '2', '2': 'Z', 'c': 'e', 'e': 'c',
            'o': 'a', 'a': 'o', 'i': 'j', 'j': 'i', 'm': 'n', 'n': 'm',
            'v': 'u', 'u': 'v', 'F': 'P', 'P': 'F'
        }

        # QWERTY keyboard adjacency for typos
        self.qwerty_map = {
            'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
            'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
            'a': 'qwsz', 's': 'qweadz', 'd': 'wserfc', 'f': 'ertdgv', 'g': 'rtyfhb',
            'h': 'tygjnm', 'j': 'yhuikm', 'k': 'uijolm', 'l': 'iopk',
            'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
            'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
        }

        # speech fillers (for ASR/transcription simulation)
        self.speech_fillers = ["um", "uh", "like", "you know", "er", "ah", "i mean"]

    def apply(self, text, method_name, rate=0.1):
        """Apply a perturbation method to text."""
        if rate == 0:
            return text

        method_name = method_name.lower()
        # print(f"applying {method_name} at rate {rate}")
        if method_name == "typos":
            return self._add_typos(text, rate)
        elif method_name == "ocr":
            return self._add_ocr(text, rate)
        elif method_name == "qwerty":
            return self._add_qwerty(text, rate)
        elif method_name == "whitespace":
            # changed: now only token-boundary disruption, no case flipping
            return self._add_whitespace_only(text, rate)
        elif method_name == "case":
            # new: isolated case randomization
            return self._add_case_only(text, rate)
        elif method_name == "whitespace_case":
            # legacy: original combined behaviour (kept for reproducibility)
            return self._add_whitespace_case_combined(text, rate)
        elif method_name == "homophones":
            return self._add_homophones(text, rate)
        elif method_name == "speech":
            return self._add_speech_fillers(text, rate)
        elif method_name == "char_insert":
            return self._add_char_insert(text, rate)
        elif method_name == "char_swap":
            return self._add_char_swap(text, rate)
        else:
            raise ValueError(f"Unknown perturbation method: {method_name}")

    # Surface Level perturbations

    def _add_typos(self, text, error_rate):
        chars = list(text)
        for i in range(len(chars) - 1, -1, -1):
            if chars[i].isdigit() or chars[i] in string.whitespace:
                continue  # don't corrupt digits or whitespace
            if random.random() < error_rate:
                r = random.random()
                if r < 0.25 and i < len(chars) - 1 and not chars[i + 1].isdigit():
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]   # swap
                elif r < 0.50:
                    del chars[i]                                        # deletion
                elif r < 0.75:
                    chars[i] = random.choice(string.ascii_lowercase)   # substitution
                else:
                    chars.insert(i, random.choice(string.ascii_lowercase))  # insertion
        return "".join(chars)

    def _add_ocr(self, text, error_rate):
        chars = list(text)
        for i in range(len(chars)):
            if chars[i] in self.ocr_map and random.random() < error_rate:
                chars[i] = self.ocr_map[chars[i]]
        return "".join(chars)

    def _add_qwerty(self, text, error_rate):
        chars = list(text)
        for i in range(len(chars)):
            char_lower = chars[i].lower()
            if char_lower in self.qwerty_map and random.random() < error_rate:
                neighbor = random.choice(self.qwerty_map[char_lower])
                chars[i] = neighbor.upper() if chars[i].isupper() else neighbor
        return "".join(chars)

    # Token-Boundary Level

    def _add_whitespace_only(self, text, error_rate):
        """
        Isolated whitespace perturbation: only injects or deletes spaces.

        This was split out from the combined whitespace_case method after
        we realized we couldn't tell which effect was causing the failures.
        """
        chars = list(text)
        for i in range(len(chars) - 1, 0, -1):
            if random.random() < error_rate:
                if chars[i] == ' ':
                    # delete an existing space (merges two tokens)
                    if random.random() < 0.5:
                        del chars[i]
                else:
                    # insert a space mid-word (splits a token)
                    if random.random() < 0.5:
                        chars.insert(i, ' ')
        return "".join(chars)

    def _add_case_only(self, text, error_rate):
        """
        Isolated case perturbation: randomly flips character case.

        Most subword tokenizers (BPE) treat 'Dog' and 'dog' as different tokens.
        """
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < error_rate:
                if chars[i].islower():
                    chars[i] = chars[i].upper()
                elif chars[i].isupper():
                    chars[i] = chars[i].lower()
        return "".join(chars)

    def _add_whitespace_case_combined(self, text, error_rate):
        """
        Legacy combined method: whitespace + case applied together.
        Retained for reproducibility of earlier experiments only.
        For new experiments use 'whitespace' and 'case' separately.
        """
        # whitespace pass
        chars = list(text)
        for i in range(len(chars) - 1, 0, -1):
            if random.random() < error_rate:
                if chars[i] == ' ':
                    if random.random() < 0.5:
                        del chars[i]
                else:
                    if random.random() < 0.5:
                        chars.insert(i, ' ')

        # case pass
        result = list("".join(chars))
        for i in range(len(result)):
            if random.random() < error_rate:
                if result[i].islower():
                    result[i] = result[i].upper()
                elif result[i].isupper():
                    result[i] = result[i].lower()
        return "".join(result)

    # Semantic Level

    def _add_homophones(self, text, error_rate):
        def replace_match(m):
            word = m.group(0)
            lower = word.lower()
            if lower not in self.homophones_map or random.random() >= error_rate:
                return word
            choice = random.choice(self.homophones_map[lower])
            # preserve original casing
            if word.isupper():
                return choice.upper()
            if word[0].isupper():
                return choice.capitalize()
            return choice
        return re.sub(r"\b[A-Za-z']+\b", replace_match, text)

    def _add_speech_fillers(self, text, error_rate):
        """Insert speech fillers (um, uh, like) to simulate ASR transcription."""
        words = text.split()
        new_words = []
        for word in words:
            new_words.append(word)
            if random.random() < error_rate:
                new_words.append(random.choice(self.speech_fillers))
        return " ".join(new_words)

    # Held-out perturbations (added 2026-05-23 for Phase 2.4 LRD eval).
    # Deliberately isolated single-op variants of typos so they are materially
    # distinct from the main-analysis perturbation set (typos mixes 4 ops at
    # 25% each; these isolate each op).

    def _add_char_insert(self, text, error_rate):
        """Random single-character insertion only (no delete/swap/substitute)."""
        chars = list(text)
        for i in range(len(chars) - 1, -1, -1):
            if chars[i].isdigit() or chars[i] in string.whitespace:
                continue
            if random.random() < error_rate:
                chars.insert(i, random.choice(string.ascii_lowercase))
        return "".join(chars)

    def _add_char_swap(self, text, error_rate):
        """Adjacent-character swap only (no other ops). Skips digits/whitespace boundaries."""
        chars = list(text)
        for i in range(len(chars) - 2, -1, -1):
            if chars[i].isdigit() or chars[i] in string.whitespace:
                continue
            if chars[i + 1].isdigit() or chars[i + 1] in string.whitespace:
                continue
            if random.random() < error_rate:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
