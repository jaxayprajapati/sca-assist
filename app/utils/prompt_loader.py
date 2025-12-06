"""
Prompt Loader Utility.

This module provides functionality to load prompts from YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from app.utils.logging import logger


class PromptLoader:
    """Utility class for loading prompts from YAML files."""

    # Path to the prompt_config directory
    PROMPT_CONFIG_DIR = Path(__file__).parent.parent / "prompt_config"

    # Cache for loaded prompts
    _cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def load_yaml(cls, filename: str) -> Dict[str, Any]:
        """
        Load a YAML file from the prompt_config directory.

        Args:
            filename: Name of the YAML file (with or without .yaml extension)

        Returns:
            Dictionary containing the YAML content
        """
        # Add .yaml extension if not present
        if not filename.endswith(".yaml"):
            filename = f"{filename}.yaml"

        # Check cache first
        if filename in cls._cache:
            return cls._cache[filename]

        # Build file path
        file_path = cls.PROMPT_CONFIG_DIR / filename

        if not file_path.exists():
            logger.error(f"Prompt config file not found: {file_path}")
            raise FileNotFoundError(f"Prompt config file not found: {file_path}")

        # Load YAML file
        with open(file_path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)

        # Cache the content
        cls._cache[filename] = content
        logger.debug(f"Loaded prompt config: {filename}")

        return content

    @classmethod
    def get_prompt(
        cls,
        filename: str,
        prompt_key: str,
        prompt_type: str = "system"
    ) -> str:
        """
        Get a specific prompt from a YAML file.

        Args:
            filename: Name of the YAML file
            prompt_key: Key for the prompt group (e.g., 'reranker', 'qa')
            prompt_type: Type of prompt ('system' or 'user')

        Returns:
            The prompt string
        """
        content = cls.load_yaml(filename)

        if prompt_key not in content:
            raise KeyError(f"Prompt key '{prompt_key}' not found in {filename}")

        prompt_group = content[prompt_key]

        if prompt_type not in prompt_group:
            raise KeyError(f"Prompt type '{prompt_type}' not found in {filename}.{prompt_key}")

        return prompt_group[prompt_type].strip()

    @classmethod
    def get_prompts(
        cls,
        filename: str,
        prompt_key: str
    ) -> Dict[str, str]:
        """
        Get all prompts for a specific key from a YAML file.

        Args:
            filename: Name of the YAML file
            prompt_key: Key for the prompt group

        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        content = cls.load_yaml(filename)

        if prompt_key not in content:
            raise KeyError(f"Prompt key '{prompt_key}' not found in {filename}")

        return content[prompt_key]

    @classmethod
    def get_config(
        cls,
        filename: str,
        prompt_key: str,
        config_key: str,
        default: Optional[Any] = None
    ) -> Any:
        """
        Get a configuration value from a prompt group.

        Args:
            filename: Name of the YAML file
            prompt_key: Key for the prompt group
            config_key: Key for the config value (e.g., 'model', 'temperature')
            default: Default value if not found

        Returns:
            The configuration value
        """
        content = cls.load_yaml(filename)

        if prompt_key not in content:
            return default

        return content[prompt_key].get(config_key, default)

    @classmethod
    def format_prompt(
        cls,
        filename: str,
        prompt_key: str,
        prompt_type: str = "user",
        **kwargs
    ) -> str:
        """
        Get and format a prompt with variables.

        Args:
            filename: Name of the YAML file
            prompt_key: Key for the prompt group
            prompt_type: Type of prompt ('system' or 'user')
            **kwargs: Variables to format into the prompt

        Returns:
            Formatted prompt string
        """
        prompt = cls.get_prompt(filename, prompt_key, prompt_type)
        return prompt.format(**kwargs)

    @classmethod
    def clear_cache(cls):
        """Clear the prompt cache."""
        cls._cache.clear()
        logger.debug("Prompt cache cleared")


# Convenience instance
prompt_loader = PromptLoader()
