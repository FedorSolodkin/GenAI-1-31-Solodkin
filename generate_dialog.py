# Сначала установите библиотеки в терминале:
# pip install transformers accelerate huggingface_hub torch     pip install sentencepiece


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import logging
from typing import Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Проверка доступности GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

class Generator:
    """Класс для генерации диалогов с использованием языковой модели"""

    def __init__(self, model_name: str = "sambanovasystems/SambaLingo-Russian-Chat"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = device

    def load_model(self) -> bool:
        """Загрузка модели и токенизатора"""
        try:
            # Авторизация в Hugging Face
            login("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")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype="auto",
                low_cpu_mem_usage=True
            )
            
            if hasattr(self.model, 'tie_weights'):
                self.model.tie_weights()
                
            logger.info("Модель успешно загружена на устройство: %s", self.device)
            return True

        except Exception as e:
            logger.error("Ошибка при загрузке модели: %s", str(e))
            return False

    def generate_dialog(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> Optional[str]:
        """Генерация диалога на основе промпта"""
        if not self.model or not self.tokenizer:
            logger.error("Модель не загружена")
            return None

        try:
            # Форматирование промпта
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"

            # Кодирование входных данных
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)

            # Генерация текста
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Декодирование результата
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Извлечение ответа модели
            response = generated_text.split("<|assistant|>")[-1].strip()
            return response

        except Exception as e:
            logger.error("Ошибка при генерации текста: %s", str(e))
            return None

def format_dialog(dialog: str) -> str:
    """Форматирование диалога для вывода"""
    lines = []
    for line in dialog.split('\n'):
        if line.strip() and not line.strip().startswith('<|'):
            lines.append(line.strip())
    return '\n'.join(lines)

def run_dialog_generation(prompt: str, **kwargs):
    """Функция для генерации и отображения диалога"""
    # Инициализация и загрузка модели
    generator = Generator()
    if not generator.load_model():
        print("Не удалось загрузить модель")
        return

    # Генерация диалога
    dialog = generator.generate_dialog(prompt, **kwargs)

    if not dialog:
        print("Не удалось сгенерировать диалог")
        return

    # Вывод результата
    print("\n" + "="*50)
    print(f"Диалог на тему: {prompt.upper()}")
    print("="*50)
    print(format_dialog(dialog))

    # Очистка памяти
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Пример использования
if __name__ == "__main__":
    # Параметры генерации
    prompt_text = "Диалог между котом и собакой о еде"
    generation_params = {
        "max_new_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1
    }

    # Запуск генерации
    run_dialog_generation(prompt_text, **generation_params)