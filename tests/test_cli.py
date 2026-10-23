"""Unit tests for HelixMind CLI and interactive console features."""

import sys
from unittest.mock import MagicMock, patch
import pytest

from helixmind.cli import (
    cmd_status,
    fetch_installed_models,
    get_system_status_panel,
    main,
    run_tool_clinical_simulation,
    run_tool_clinvar,
    run_tool_crispr,
    run_tool_denovo,
    run_tool_depmap,
    run_tool_openfda,
    run_tool_tme,
)
from helixmind.llm.ollama_client import OllamaClient


def test_system_status_panel():
    client = OllamaClient()
    panel = get_system_status_panel(client)
    assert panel is not None
    assert "HelixMind System Configuration" in str(panel.title)


def test_fetch_installed_models():
    client = OllamaClient()
    with patch.object(client, "list_models", return_value=[{"name": "llama3.2:1b"}]):
        models = fetch_installed_models(client)
        assert len(models) == 1
        assert models[0]["name"] == "llama3.2:1b"


def test_cmd_status():
    client = OllamaClient()
    with patch.object(client, "check_health", return_value=True):
        with patch.object(client, "list_models", return_value=[{"name": "llama3.2:1b", "size": 1024**3}]):
            # Should run without error
            cmd_status(client)


def test_standalone_in_silico_tools():
    # 1. DepMap
    run_tool_depmap("EGFR", "Glioblastoma")

    # 2. CRISPR
    run_tool_crispr("CD274")

    # 3. openFDA
    run_tool_openfda("tamoxifen")

    # 4. ClinVar
    run_tool_clinvar("EGFR", "Glioblastoma")

    # 5. Single-Cell TME
    run_tool_tme("STAT3,CD274", "Glioblastoma")

    # 6. De Novo Analogs
    run_tool_denovo("tamoxifen")

    # 7. Virtual Clinical Trial
    run_tool_clinical_simulation("Trastuzumab", "Breast Cancer")


def test_cli_config_command(capsys):
    with patch.object(sys, "argv", ["helixmind", "config", "--list"]):
        main()


def test_cli_tools_subcommand():
    with patch.object(sys, "argv", ["helixmind", "tools", "depmap", "EGFR"]):
        main()

    with patch.object(sys, "argv", ["helixmind", "tools", "crispr", "CD274"]):
        main()


def test_cli_background_commands():
    with patch.object(sys, "argv", ["helixmind", "ps"]):
        main()

    with patch.object(sys, "argv", ["helixmind", "logs", "NONEXISTENT", "-n", "10"]):
        main()

    with patch.object(sys, "argv", ["helixmind", "stop", "NONEXISTENT"]):
        main()
