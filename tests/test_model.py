from pathlib import Path
import pytest
import torch

# Импортируйте ваш загрузчик и модель из open_mythos
# from open_mythos.config import ModelConfig
# from open_mythos.model import MythosModel

CONFIG_PATH = Path("configs/model_micro_1.5b.json")

def test_full_model_forward_pass():
    # Загружаем параметры строго из тестового JSON-конфига проекта
    config = ModelConfig.from_json(CONFIG_PATH)
    model = MythosModel(config)
    model.eval()

    # Фиктивный вход на основе параметров из загруженного конфига
    dummy_input = torch.randint(0, config.vocab_size, (1, 8))

    with torch.no_grad():
        output = model(dummy_input)

    assert output is not None
