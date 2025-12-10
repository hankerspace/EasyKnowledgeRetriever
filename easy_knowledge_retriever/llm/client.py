from typing import Any
from easy_knowledge_retriever.utils.logger import logger
from openai import AsyncOpenAI

def create_openai_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
    use_azure: bool = False,
    azure_deployment: str | None = None,
    api_version: str | None = None,
    timeout: int | None = None,
    client_configs: dict[str, Any] | None = None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI or AsyncAzureOpenAI client with the given configuration.

    Args:
        api_key: OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        base_url: Base URL for the OpenAI API. If None, uses the default OpenAI API URL.
        use_azure: Whether to create an Azure OpenAI client. Default is False.
        azure_deployment: Azure OpenAI deployment name (only used when use_azure=True).
        api_version: Azure OpenAI API version (only used when use_azure=True).
        timeout: Request timeout in seconds.
        client_configs: Additional configuration options for the AsyncOpenAI client.
            These will override any default configurations but will be overridden by
            explicit parameters (api_key, base_url).

    Returns:
        An AsyncOpenAI or AsyncAzureOpenAI client instance.
    """
    if use_azure:
        from openai import AsyncAzureOpenAI

        if not api_key:
            raise ValueError("Azure OpenAI API key must be provided")

        if client_configs is None:
            client_configs = {}

        # Create a merged config dict with precedence: explicit params > client_configs
        merged_configs = {
            **client_configs,
            "api_key": api_key,
        }

        # Add explicit parameters (override client_configs)
        if base_url is not None:
            merged_configs["azure_endpoint"] = base_url
        if azure_deployment is not None:
            merged_configs["azure_deployment"] = azure_deployment
        if api_version is not None:
            merged_configs["api_version"] = api_version
        if timeout is not None:
            merged_configs["timeout"] = timeout

        return AsyncAzureOpenAI(**merged_configs)
    else:
        if not api_key:
             # raise ValueError("OpenAI API key must be provided")
             pass # Let OpenAI client handle missing key if it wants, or it will error.

        default_headers = {
            "User-Agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) EasyKnowledgeRetriever",
            "Content-Type": "application/json",
        }

        if client_configs is None:
            client_configs = {}

        # Create a merged config dict with precedence: explicit params > client_configs > defaults
        merged_configs = {
            **client_configs,
            "default_headers": default_headers,
            "api_key": api_key,
        }

        if base_url is not None:
            merged_configs["base_url"] = base_url

        if timeout is not None:
            merged_configs["timeout"] = timeout

        return AsyncOpenAI(**merged_configs)
