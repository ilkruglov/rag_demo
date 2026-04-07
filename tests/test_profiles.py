from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.config as config


def _use_temp_runtime(monkeypatch, tmp_path: Path) -> Path:
    runtime_path = tmp_path / "runtime_settings.json"
    monkeypatch.setattr(config, "RUNTIME_CONFIG_PATH", runtime_path)
    config.reload_settings()
    return runtime_path


def test_named_profile_resolves_profile_paths_and_prompt(monkeypatch, tmp_path):
    runtime_path = _use_temp_runtime(monkeypatch, tmp_path)
    runtime_path.write_text(
        """
        {
          "active_profile": "support-faq",
          "profiles": {
            "support-faq": {
              "label": "Support FAQ",
              "system_prompt": "Support prompt",
              "retrieval_top_k": 7
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    settings = config.get_settings()

    assert settings.active_profile == "support-faq"
    assert settings.system_prompt == "Support prompt"
    assert settings.retrieval_top_k == 7
    assert settings.qdrant_path == "data/profiles/support-faq/qdrant"
    assert settings.storage_dir == "data/profiles/support-faq/storage"
    assert config.get_raw_documents_dir(settings) == Path("data/profiles/support-faq/raw")


def test_save_runtime_settings_keeps_default_and_named_profiles_separate(monkeypatch, tmp_path):
    runtime_path = _use_temp_runtime(monkeypatch, tmp_path)

    config.save_runtime_settings(
        {
            "system_prompt": "Default prompt",
            "retrieval_top_k": 11,
        },
        profile_id="default",
    )
    config.save_runtime_settings(
        {
            "label": "Support",
            "system_prompt": "Support prompt",
            "retrieval_top_k": 5,
        },
        profile_id="support",
    )

    default_settings = config.get_settings("default")
    support_settings = config.get_settings("support")
    saved = runtime_path.read_text(encoding="utf-8")

    assert default_settings.system_prompt == "Default prompt"
    assert default_settings.retrieval_top_k == 11
    assert support_settings.system_prompt == "Support prompt"
    assert support_settings.retrieval_top_k == 5
    assert support_settings.qdrant_collection == "support_documents"
    assert '"support"' in saved
    assert '"Default prompt"' in saved
    assert '"Support prompt"' in saved


def test_delete_profile_resets_active_profile(monkeypatch, tmp_path):
    _use_temp_runtime(monkeypatch, tmp_path)

    config.save_profile(
        "ops-review",
        {
            "label": "Ops Review",
            "system_prompt": "Ops prompt",
        },
        activate=True,
    )

    assert config.get_settings().active_profile == "ops-review"

    active_profile = config.delete_profile("ops-review")

    assert active_profile == "default"
    assert config.get_settings().active_profile == "default"


def test_default_profile_label_is_sdek(monkeypatch, tmp_path):
    _use_temp_runtime(monkeypatch, tmp_path)

    catalog = config.get_profile_catalog()

    assert catalog["default"]["label"] == "СДЭК"
    assert catalog["default"]["description"] == "Базовый профиль FAQ СДЭК."
