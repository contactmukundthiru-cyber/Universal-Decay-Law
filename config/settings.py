"""
Configuration settings for the Universal Decay Law project.
Uses pydantic-settings for type-safe configuration management.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    model_config = SettingsConfigDict(env_prefix="DB_")

    driver: str = Field(default="sqlite+aiosqlite", description="Database driver")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="universal_decay_law", description="Database name")
    user: str = Field(default="", description="Database user")
    password: str = Field(default="", description="Database password")
    sqlite_path: str = Field(default="./database/decay_law.db", description="SQLite database path")

    @property
    def url(self) -> str:
        """Generate database URL."""
        if "sqlite" in self.driver:
            return f"{self.driver}:///{self.sqlite_path}"
        return f"{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration for caching and task queue."""
    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")

    @property
    def url(self) -> str:
        """Generate Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class APIKeysSettings(BaseSettings):
    """API keys for various platforms."""
    model_config = SettingsConfigDict(env_prefix="API_")

    # Reddit
    reddit_client_id: str = Field(default="", description="Reddit API client ID")
    reddit_client_secret: str = Field(default="", description="Reddit API client secret")
    reddit_user_agent: str = Field(
        default="UniversalDecayLaw/1.0",
        description="Reddit API user agent"
    )

    # GitHub
    github_token: str = Field(default="", description="GitHub personal access token")

    # Strava
    strava_client_id: str = Field(default="", description="Strava API client ID")
    strava_client_secret: str = Field(default="", description="Strava API client secret")

    # Last.fm
    lastfm_api_key: str = Field(default="", description="Last.fm API key")
    lastfm_shared_secret: str = Field(default="", description="Last.fm shared secret")


class AnalysisSettings(BaseSettings):
    """Analysis configuration parameters."""
    model_config = SettingsConfigDict(env_prefix="ANALYSIS_")

    # Time windows
    max_observation_days: int = Field(default=365, description="Maximum observation window in days")
    min_activity_threshold: int = Field(default=5, description="Minimum activities to include user")

    # Model fitting
    default_decay_model: str = Field(
        default="stretched_exponential",
        description="Default decay model (stretched_exponential, power_law, weibull, double_exponential)"
    )
    mcmc_samples: int = Field(default=5000, description="Number of MCMC samples")
    mcmc_chains: int = Field(default=4, description="Number of MCMC chains")
    mcmc_tune: int = Field(default=1000, description="Number of tuning samples")

    # Validation
    train_test_split: float = Field(default=0.8, description="Train/test split ratio")
    cross_validation_folds: int = Field(default=5, description="Number of cross-validation folds")

    # Normalization
    normalization_method: str = Field(
        default="zscore",
        description="Normalization method (zscore, minmax, robust)"
    )


class ServerSettings(BaseSettings):
    """Server configuration."""
    model_config = SettingsConfigDict(env_prefix="SERVER_")

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    workers: int = Field(default=4, description="Number of workers")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated CORS origins"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


class Settings(BaseSettings):
    """Main settings class combining all configurations."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project info
    project_name: str = Field(
        default="Universal Decay Law",
        description="Project name"
    )
    version: str = Field(default="1.0.0", description="Project version")
    environment: str = Field(default="development", description="Environment (development, production)")

    # JWT settings for dashboard auth
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="JWT secret key"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Token expiration time")

    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api_keys: APIKeysSettings = Field(default_factory=APIKeysSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export settings instance
settings = get_settings()
