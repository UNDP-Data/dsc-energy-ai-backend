"""Tests for copyable assistant kit validation and installation."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from src.rag_system import AssistantKitError, default_table_names, validate_kit, validate_manifest_file

ROOT = Path(__file__).resolve().parents[1]


def test_template_assistant_kit_validates():
    kit = validate_kit(ROOT / "assistant_kits" / "_template")
    assert kit.profile.assistant_id == "example_assistant"
    assert kit.profile.table_names == default_table_names("example_assistant")
    assert kit.manifest_path is not None


def test_sample_assistant_kit_validates():
    kit = validate_kit(ROOT / "assistant_kits" / "sample")
    assert kit.profile.assistant_id == "sample"
    assert kit.profile.table_names == {
        "chunks": "sample_chunks",
        "documents": "sample_documents",
        "sources": "sample_sources",
    }
    assert kit.eval_path is not None


def test_kit_rejects_unsafe_assistant_id(tmp_path):
    kit_dir = tmp_path / "badkit"
    (kit_dir / "corpus").mkdir(parents=True)
    profile = yaml.safe_load((ROOT / "assistant_kits" / "_template" / "assistant.yaml").read_text())
    profile["assistant_id"] = "../bad"
    (kit_dir / "assistant.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    manifest = yaml.safe_load((ROOT / "assistant_kits" / "_template" / "corpus" / "manifest.yaml").read_text())
    manifest["assistant_id"] = "../bad"
    (kit_dir / "corpus" / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(AssistantKitError, match="assistant_id"):
        validate_kit(kit_dir)


def test_kit_rejects_nonstandard_tables_for_non_sea(tmp_path):
    kit_dir = tmp_path / "badkit"
    (kit_dir / "corpus").mkdir(parents=True)
    profile = yaml.safe_load((ROOT / "assistant_kits" / "_template" / "assistant.yaml").read_text())
    profile["assistant_id"] = "badkit"
    profile["tables"] = {"chunks": "chunks", "documents": "documents", "sources": "sources"}
    (kit_dir / "assistant.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    manifest = yaml.safe_load((ROOT / "assistant_kits" / "_template" / "corpus" / "manifest.yaml").read_text())
    manifest["assistant_id"] = "badkit"
    (kit_dir / "corpus" / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(AssistantKitError, match="standard namespace"):
        validate_kit(kit_dir)


def test_manifest_validation_requires_chunk_or_content(tmp_path):
    manifest = yaml.safe_load((ROOT / "assistant_kits" / "sample" / "corpus" / "manifest.yaml").read_text())
    manifest["documents"][0].pop("chunks", None)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(
        AssistantKitError,
        match="content, non-empty chunks, or chunk_count with manifest.chunk_files",
    ):
        validate_manifest_file(manifest_path, expected_assistant_id="sample")


def test_validate_assistant_kit_script_outputs_json():
    result = subprocess.run(
        [sys.executable, "scripts/validate_assistant_kit.py", "--kit", "assistant_kits/sample"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"ok": true' in result.stdout.lower()
    assert '"assistant_id": "sample"' in result.stdout


def test_install_assistant_kit_script_installs_to_temp_dirs(tmp_path):
    profiles_dir = tmp_path / "profiles"
    kits_dir = tmp_path / "kits"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/install_assistant_kit.py",
            "--kit",
            "assistant_kits/sample",
            "--profiles-dir",
            str(profiles_dir),
            "--kits-dir",
            str(kits_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"ok": true' in result.stdout.lower()
    assert (profiles_dir / "sample.yaml").exists()
    assert (kits_dir / "sample" / "assistant.yaml").exists()
    installed_profile = yaml.safe_load((profiles_dir / "sample.yaml").read_text())
    assert installed_profile["tables"]["chunks"] == "sample_chunks"


def test_export_assistant_kit_script_copies_template_to_target(tmp_path):
    target = tmp_path / "other_repo" / "my_assistant_kit"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_assistant_kit.py",
            "--kit",
            "assistant_kits/_template",
            "--target",
            str(target),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"ok": true' in result.stdout.lower()
    assert (target / "assistant.yaml").exists()
    assert (target / "corpus" / "manifest.yaml").exists()
    assert (target / "CODEX_HANDOFF.md").exists()
    copied_profile = yaml.safe_load((target / "assistant.yaml").read_text())
    assert copied_profile["tables"]["chunks"] == "example_assistant_chunks"


def test_deploy_assistant_kit_script_can_run_from_other_repo(tmp_path):
    other_repo = tmp_path / "other_repo"
    source_kit = other_repo / "sgp_ai"
    profiles_dir = tmp_path / "backend_profiles"
    kits_dir = tmp_path / "backend_kits"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_assistant_kit.py",
            "--kit",
            "assistant_kits/sample",
            "--target",
            str(source_kit),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "deploy_assistant_kit.py"),
            "--kit",
            "sgp_ai",
            "--profiles-dir",
            str(profiles_dir),
            "--kits-dir",
            str(kits_dir),
        ],
        cwd=other_repo,
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"ok": true' in result.stdout.lower()
    assert (profiles_dir / "sample.yaml").exists()
    assert (kits_dir / "sample" / "assistant.yaml").exists()
    payload = yaml.safe_load((profiles_dir / "sample.yaml").read_text())
    assert payload["assistant_id"] == "sample"
