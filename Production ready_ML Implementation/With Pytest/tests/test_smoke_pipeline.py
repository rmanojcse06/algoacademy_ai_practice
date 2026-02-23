from src.components.model_trainer.train import MODELS

def test_multiple_models_available():
    assert len(MODELS) >= 2