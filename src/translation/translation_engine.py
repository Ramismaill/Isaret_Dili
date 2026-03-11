import requests
import json
from collections import deque


class TranslationEngine:
    """
    GRU'dan gelen gloss dizilerini alarak
    Ollama üzerinden yerel LLM ile doğal Türkçe
    cümlelere dönüştüren sınıf.
    """

    def __init__(self, model_name="phi4-mini",
                 ollama_url="http://localhost:11434",
                 window_size=5):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.window_size = window_size
        self.gloss_buffer = deque(maxlen=window_size)
        self.translation_history = []

        self.system_prompt = """You are an expert Sign Language Translator 
specializing in Turkish Sign Language (TİD). Your task is to translate 
raw, ungrammatical sequence of words (glosses) derived from a neural 
network into natural, grammatically correct, and fluent conversational 
Turkish sentences. Do not add extra information that is not implied. 
Fill in the missing morphological cases and adjust the word order.

Examples:
Input: "BEN GİTMEK OKUL YARIN" -> Output: "Ben yarın okula gideceğim."
Input: "BABA ARABA ALMAK YENİ" -> Output: "Babam yeni bir araba aldı."
Input: "SU İSTEMEK" -> Output: "Su istiyorum."
"""

    def add_gloss(self, gloss: str):
        """Buffer'a yeni gloss kelimesi ekler."""
        self.gloss_buffer.append(gloss.upper())

    def translate(self) -> str:
        """Buffer'daki gloss dizisini Türkçe'ye çevirir."""
        if not self.gloss_buffer:
            return ""

        gloss_sequence = " ".join(self.gloss_buffer)
        return self._call_ollama(gloss_sequence)

    def translate_direct(self, gloss_sequence: str) -> str:
        """Direkt gloss dizisi alarak çeviri yapar."""
        return self._call_ollama(gloss_sequence)

    def _call_ollama(self, gloss_sequence: str) -> str:
        """Ollama API'sine istek gönderir."""
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",
                     "content": f"Translate: '{gloss_sequence}'"}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 100
                }
            }

            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                translation = result['message']['content'].strip()
                self.translation_history.append({
                    "gloss": gloss_sequence,
                    "translation": translation
                })
                return translation
            else:
                return f"[Hata: {response.status_code}]"

        except requests.exceptions.ConnectionError:
            return "[Ollama bağlantı hatası - sunucu çalışıyor mu?]"
        except Exception as e:
            return f"[Hata: {str(e)}]"

    def clear_buffer(self):
        self.gloss_buffer.clear()

    def is_ollama_running(self) -> bool:
        """Ollama sunucusunun çalışıp çalışmadığını kontrol eder."""
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags", timeout=5
            )
            return response.status_code == 200
        except:
            return False