from medvision.config import get_device, get_labels_map, get_settings


def test_settings_load_defaults():
    settings = get_settings()
    assert settings.vision.weights == "densenet121-res224-all"
    assert settings.kg.reasoning.top_k_differentials == 3
    assert settings.disclaimer  # never empty


def test_labels_map_has_entries():
    labels = get_labels_map()
    assert "Effusion" in labels
    assert labels["Effusion"] == "F_EFFUSION"
    assert len(labels) == 18


def test_get_device_returns_valid_choice():
    device = get_device()
    assert device in {"mps", "cuda", "cpu"}
