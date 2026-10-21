"""Configuration settings for HelixMind platform."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the Ollama API server",
    )
    ollama_timeout: float = Field(
        default=120.0,
        description="Timeout in seconds for Ollama requests",
    )
    ollama_default_model: str = Field(
        default="llama3.2:1b",
        description="Default general-purpose LLM for agents",
    )
    ollama_reasoning_model: str = Field(
        default="llama3.2:1b",
        description="Reasoning/thinking model for hypothesis formulation",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text:latest",
        description="Embedding model for scientific papers and semantic search",
    )

    # Scientific Research Defaults
    default_max_hypotheses: int = Field(
        default=3,
        description="Maximum hypotheses to generate per investigation cycle",
    )
    default_literature_limit: int = Field(
        default=5,
        description="Number of papers to fetch and analyze per topic",
    )
    default_trials_limit: int = Field(
        default=4,
        description="Number of clinical trials to fetch and analyze per topic",
    )
    default_depth: str = Field(
        default="deep",
        description="Default investigation depth: deep, exhaustive, or institutional",
    )

    # In-Silico & Rigor Module Toggles
    enable_depmap: bool = Field(
        default=True,
        description="Enable DepMap cancer dependency verification",
    )
    enable_pharmacovigilance: bool = Field(
        default=True,
        description="Enable openFDA FAERS adverse reaction surveillance",
    )
    enable_clinvar: bool = Field(
        default=True,
        description="Enable NCBI ClinVar genomic stratification",
    )
    enable_tme: bool = Field(
        default=True,
        description="Enable Single-Cell TME interactome mapping",
    )
    enable_crispr: bool = Field(
        default=True,
        description="Enable CRISPR sgRNA and primer design",
    )
    enable_tribunal: bool = Field(
        default=True,
        description="Enable Adversarial Peer-Review Tribunal",
    )
    export_formats: str = Field(
        default="markdown,pdf,latex,bibtex",
        description="Comma-separated default export formats (markdown, pdf, latex, bibtex)",
    )

    # Biomedical APIs Configuration
    ncbi_api_key: str = Field(
        default="1742f98e7b166a1aa284df7b00c9ae14ba09",
        description="API Key for NCBI Entrez E-utilities",
    )
    ncbi_email: str = Field(
        default="helixmind-research@example.com",
        description="Email identifier for NCBI requests",
    )
    ncbi_tool: str = Field(
        default="helixmind-platform",
        description="Tool identifier for NCBI requests",
    )
    ncbi_base_url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        description="NCBI E-utilities base URL",
    )
    clinical_trials_base_url: str = Field(
        default="https://clinicaltrials.gov/api/v2",
        description="ClinicalTrials.gov API v2 base URL",
    )
    europe_pmc_base_url: str = Field(
        default="https://www.ebi.ac.uk/europepmc/webservices/rest",
        description="Europe PMC REST API base URL",
    )
    europe_pmc_annotations_url: str = Field(
        default="https://www.ebi.ac.uk/europepmc/annotations_api",
        description="Europe PMC Annotations API base URL",
    )

    # Storage paths
    cache_dir: Path = Field(
        default=Path(".cache"),
        description="Local cache directory for fetched papers and responses",
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for platform datasets and persistent records",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    settings = Settings()
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings


def save_env_variables(updates: Dict[str, Any], env_path: Path = Path(".env")) -> None:
    """Safely update or add key-value pairs in the .env file."""
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    str_updates = {k.upper(): str(v) for k, v in updates.items()}
    updated_keys = set()

    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip().upper()
            if key in str_updates:
                new_lines.append(f"{key}={str_updates[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    for k, v in str_updates.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def update_settings(updates: Dict[str, Any], persist: bool = True, env_path: Path = Path(".env")) -> Settings:
    """Update settings in environment and clear cached singleton. Optionally persists to .env."""
    for k, v in updates.items():
        os.environ[k.upper()] = str(v)

    if persist:
        save_env_variables(updates, env_path=env_path)

    get_settings.cache_clear()
    return get_settings()
