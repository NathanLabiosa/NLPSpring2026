import random
import string
import re

class PerturbationEngine:
    def __init__(self):
        # Semantic: Homophones map
        self.homophones_map = {
            "their": ["there", "they're"], "there": ["their", "they're"], "they're": ["their", "there"],
            "your": ["you're"], "you're": ["your"], "its": ["it's"], "it's": ["its"],
            "to": ["too", "two"], "too": ["to", "two"], "two": ["to", "too"],
            "then": ["than"], "than": ["then"], "weather": ["whether"], "whether": ["weather"],
            "write": ["right"], "right": ["write"], "read": ["red"], "red": ["read"],
            "for": ["four"], "four": ["for"], "sun": ["son"], "son": ["sun"]
        }
        
        # Surface: OCR visual lookalikes
        self.ocr_map = {
            'l': '1', '1': 'l', 'I': '1', 'O': '0', '0': 'O', 'S': '5', '5': 'S',
            'B': '8', '8': 'B', 'Z': '2', '2': 'Z', 'c': 'e', 'e': 'c',
            'o': 'a', 'a': 'o', 'i': 'j', 'j': 'i', 'm': 'n', 'n': 'm',
            'v': 'u', 'u': 'v', 'F': 'P', 'P': 'F'
        }

        # Surface: QWERTY keyboard adjacency
        self.qwerty_map = {
            'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy', 'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
            'a': 'qwsz', 's': 'qweadz', 'd': 'wserfc', 'f': 'ertdgv', 'g': 'rtyfhb', 'h': 'tygjnm', 'j': 'yhuikm', 'k': 'uijolm', 'l': 'iopk',
            'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
        }

        # Semantic: Speech Fillers
        self.speech_fillers = ["um", "uh", "like", "you know", "er", "ah", "i mean"]

    def apply(self, text, method_name, rate=0.1):
        """Unified interface to apply any perturbation."""
        if rate == 0: return text
        
        method_name = method_name.lower()
        if method_name == "typos":
            return self._add_typos(text, rate)
        elif method_name == "ocr":
            return self._add_ocr(text, rate)
        elif method_name == "qwerty":
            return self._add_qwerty(text, rate)
        elif method_name == "whitespace":
            return self._add_whitespace_case(text, rate)
        elif method_name == "homophones":
            return self._add_homophones(text, rate)
        elif method_name == "speech":
            return self._add_speech_fillers(text, rate)
        else:
            raise ValueError(f"Unknown perturbation method: {method_name}")

    # --- Surface Level ---
    def _add_typos(self, text, error_rate):
        chars = list(text)
        for i in range(len(chars) - 1, -1, -1):
            if chars[i].isdigit() or chars[i] in string.whitespace: continue
            if random.random() < error_rate:
                r = random.random()
                if r < 0.25 and i < len(chars)-1 and not chars[i+1].isdigit():
                    chars[i], chars[i+1] = chars[i+1], chars[i] 
                elif r < 0.50: del chars[i] 
                elif r < 0.75: chars[i] = random.choice(string.ascii_lowercase)
                else: chars.insert(i, random.choice(string.ascii_lowercase))
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

    # --- Token Level ---
    def _add_whitespace_case(self, text, error_rate):
        chars = list(text)
        # Whitespace injection/deletion
        for i in range(len(chars) - 1, 0, -1):
            if random.random() < error_rate:
                if chars[i] == ' ':
                    if random.random() < 0.5: del chars[i]
                else:
                    if random.random() < 0.5: chars.insert(i, ' ')
        
        # Random Case injection
        result = list("".join(chars))
        for i in range(len(result)):
            if random.random() < error_rate:
                if result[i].islower(): result[i] = result[i].upper()
                elif result[i].isupper(): result[i] = result[i].lower()
        return "".join(result)

    # --- Semantic Level ---
    def _add_homophones(self, text, error_rate):
        def replace_match(m):
            word = m.group(0)
            lower = word.lower()
            if lower not in self.homophones_map or random.random() >= error_rate:
                return word
            choice = random.choice(self.homophones_map[lower])
            if word.isupper(): return choice.upper()
            if word[0].isupper(): return choice.capitalize()
            return choice
        return re.sub(r"\b[A-Za-z']+\b", replace_match, text)

    def _add_speech_fillers(self, text, error_rate):
        words = text.split()
        new_words = []
        for word in words:
            new_words.append(word)
            if random.random() < error_rate:
                new_words.append(random.choice(self.speech_fillers))
        return " ".join(new_words)