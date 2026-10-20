"""Configuration settings for HelixMind platform."""

from functools import lru_cache
from pathlib import Path
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
        default="llama3.2:3b",
        description="Default general-purpose LLM for agents",
    )
    ollama_reasoning_model: str = Field(
        default="deepseek-r1:1.5b",
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
