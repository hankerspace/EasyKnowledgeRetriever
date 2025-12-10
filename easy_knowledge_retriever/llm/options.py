"""
Module that implements containers for specific LLM bindings.

This module provides container implementations for various Large Language Model
bindings and integrations.
"""

from argparse import ArgumentParser, Namespace
import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, List, get_args, get_origin


from easy_knowledge_retriever.config.llm_config import DEFAULT_TEMPERATURE


def _resolve_optional_type(field_type: Any) -> Any:
    """Return the concrete type for Optional/Union annotations."""
    origin = get_origin(field_type)
    if origin in (list, dict, tuple):
        return field_type

    args = get_args(field_type)
    if args:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0]
    return field_type


# =============================================================================
# BindingOptions Base Class
# =============================================================================
@dataclass
class BindingOptions:
    """Base class for binding options."""

    # mandatory name of binding
    _binding_name: ClassVar[str]

    # optional help message for each option
    _help: ClassVar[dict[str, str]]

    @staticmethod
    def _all_class_vars(klass: type, include_inherited=True) -> dict[str, Any]:
        """Print class variables, optionally including inherited ones"""
        if include_inherited:
            # Get all class variables from MRO
            vars_dict = {}
            for base in reversed(klass.__mro__[:-1]):  # Exclude 'object'
                vars_dict.update(
                    {
                        k: v
                        for k, v in base.__dict__.items()
                        if (
                            not k.startswith("_")
                            and not callable(v)
                            and not isinstance(v, classmethod)
                        )
                    }
                )
        else:
            # Only direct class variables
            vars_dict = {
                k: v
                for k, v in klass.__dict__.items()
                if (
                    not k.startswith("_")
                    and not callable(v)
                    and not isinstance(v, classmethod)
                )
            }

        return vars_dict

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        group = parser.add_argument_group(f"{cls._binding_name} binding options")
        for arg_item in cls.args_env_name_type_value():
            # Handle JSON parsing for list types
            if arg_item["type"] is List[str]:

                def json_list_parser(value):
                    try:
                        parsed = json.loads(value)
                        if not isinstance(parsed, list):
                            raise argparse.ArgumentTypeError(
                                f"Expected JSON array, got {type(parsed).__name__}"
                            )
                        return parsed
                    except json.JSONDecodeError as e:
                        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")

                # Get environment variable with JSON parsing
                env_value = argparse.SUPPRESS
                if env_value is not argparse.SUPPRESS:
                    try:
                        env_value = json_list_parser(env_value)
                    except argparse.ArgumentTypeError:
                        env_value = argparse.SUPPRESS

                group.add_argument(
                    f"--{arg_item['argname']}",
                    type=json_list_parser,
                    default=env_value,
                    help=arg_item["help"],
                )
            # Handle JSON parsing for dict types
            elif arg_item["type"] is dict:

                def json_dict_parser(value):
                    try:
                        parsed = json.loads(value)
                        if not isinstance(parsed, dict):
                            raise argparse.ArgumentTypeError(
                                f"Expected JSON object, got {type(parsed).__name__}"
                            )
                        return parsed
                    except json.JSONDecodeError as e:
                        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")

                # Get environment variable with JSON parsing
                env_value = argparse.SUPPRESS
                if env_value is not argparse.SUPPRESS:
                    try:
                        env_value = json_dict_parser(env_value)
                    except argparse.ArgumentTypeError:
                        env_value = argparse.SUPPRESS

                group.add_argument(
                    f"--{arg_item['argname']}",
                    type=json_dict_parser,
                    default=env_value,
                    help=arg_item["help"],
                )
            # Handle boolean types specially to avoid argparse bool() constructor issues
            elif arg_item["type"] is bool:

                def bool_parser(value):
                    """Custom boolean parser that handles string representations correctly"""
                    if isinstance(value, bool):
                        return value
                    if isinstance(value, str):
                        return value.lower() in ("true", "1", "yes", "t", "on")
                    return bool(value)

                # Get environment variable with proper type conversion
                env_value = argparse.SUPPRESS

                group.add_argument(
                    f"--{arg_item['argname']}",
                    type=bool_parser,
                    default=env_value,
                    help=arg_item["help"],
                )
            else:
                resolved_type = arg_item["type"]
                if resolved_type is not None:
                    resolved_type = _resolve_optional_type(resolved_type)

                group.add_argument(
                    f"--{arg_item['argname']}",
                    type=resolved_type,
                    default=arg_item["default"],
                    help=arg_item["help"],
                )

    @classmethod
    def args_env_name_type_value(cls):
        import dataclasses

        args_prefix = f"{cls._binding_name}".replace("_", "-")
        env_var_prefix = f"{cls._binding_name}_".upper()
        help = cls._help

        # Check if this is a dataclass and use dataclass fields
        if dataclasses.is_dataclass(cls):
            for field in dataclasses.fields(cls):
                # Skip private fields
                if field.name.startswith("_"):
                    continue

                # Get default value
                if field.default is not dataclasses.MISSING:
                    default_value = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    default_value = field.default_factory()
                else:
                    default_value = None

                argdef = {
                    "argname": f"{args_prefix}-{field.name}",
                    "env_name": f"{env_var_prefix}{field.name.upper()}",
                    "type": _resolve_optional_type(field.type),
                    "default": default_value,
                    "help": f"{cls._binding_name} -- " + help.get(field.name, ""),
                }

                yield argdef
        else:
            # Fallback to old method for non-dataclass classes
            class_vars = {
                key: value
                for key, value in cls._all_class_vars(cls).items()
                if not callable(value) and not key.startswith("_")
            }

            # Get type hints to properly detect List[str] types
            type_hints = {}
            for base in cls.__mro__:
                if hasattr(base, "__annotations__"):
                    type_hints.update(base.__annotations__)

            for class_var in class_vars:
                # Use type hint if available, otherwise fall back to type of value
                var_type = type_hints.get(class_var, type(class_vars[class_var]))

                argdef = {
                    "argname": f"{args_prefix}-{class_var}",
                    "env_name": f"{env_var_prefix}{class_var.upper()}",
                    "type": var_type,
                    "default": class_vars[class_var],
                    "help": f"{cls._binding_name} -- " + help.get(class_var, ""),
                }

                yield argdef

    @classmethod
    def generate_dot_env_sample(cls):
        """
        Generate a sample .env file for all EasyKnowledgeRetriever binding options.

        This method creates a .env file that includes all the binding options
        defined by the subclasses of BindingOptions. It uses the args_env_name_type_value()
        method to get the list of all options and their default values.

        Returns:
            str: A string containing the contents of the sample .env file.
        """
        from io import StringIO

        sample_top = (
            "#" * 80
            + "\n"
            + (
                "# Autogenerated .env entries list for EasyKnowledgeRetriever binding options\n"
                "#\n"
                "# To generate run:\n"
                "# $ python -m easy_knowledge_retriever.llm.options\n"
            )
            + "#" * 80
            + "\n"
        )

        sample_bottom = (
            ("#\n# End of .env entries for EasyKnowledgeRetriever binding options\n")
            + "#" * 80
            + "\n"
        )

        sample_stream = StringIO()
        sample_stream.write(sample_top)
        for klass in cls.__subclasses__():
            for arg_item in klass.args_env_name_type_value():
                if arg_item["help"]:
                    sample_stream.write(f"# {arg_item['help']}\n")

                # Handle JSON formatting for list and dict types
                if arg_item["type"] is List[str] or arg_item["type"] is dict:
                    default_value = json.dumps(arg_item["default"])
                else:
                    default_value = arg_item["default"]

                sample_stream.write(f"# {arg_item['env_name']}={default_value}\n\n")

        sample_stream.write(sample_bottom)
        return sample_stream.getvalue()

    @classmethod
    def options_dict(cls, args: Namespace) -> dict[str, Any]:
        """
        Extract options dictionary for a specific binding from parsed arguments.

        This method filters the parsed command-line arguments to return only those
        that belong to the specific binding class. It removes the binding prefix
        from argument names to create a clean options dictionary.

        Args:
            args (Namespace): Parsed command-line arguments containing all binding options

        Returns:
            dict[str, Any]: Dictionary mapping option names (without prefix) to their values

        Example:
            If args contains {'num_ctx': 512, 'other_option': 'value'}
            and this is called it returns {'num_ctx': 512}
        """
        prefix = cls._binding_name + "_"
        skipchars = len(prefix)
        options = {
            key[skipchars:]: value
            for key, value in vars(args).items()
            if key.startswith(prefix)
        }

        return options

    def asdict(self) -> dict[str, Any]:
        """
        Convert an instance of binding options to a dictionary.

        This method uses dataclasses.asdict() to convert the dataclass instance
        into a dictionary representation, including all its fields and values.

        Returns:
            dict[str, Any]: Dictionary representation of the binding options instance
        """
        return asdict(self)


# =============================================================================
# Binding Options for OpenAI
# =============================================================================
@dataclass
class OpenAILLMOptions(BindingOptions):
    """Options for OpenAI LLM with configuration for OpenAI and Azure OpenAI API calls."""

    # mandatory name of binding
    _binding_name: ClassVar[str] = "openai_llm"

    # Sampling and generation parameters
    frequency_penalty: float = 0.0  # Penalty for token frequency (-2.0 to 2.0)
    max_completion_tokens: int = None  # Maximum number of tokens to generate
    presence_penalty: float = 0.0  # Penalty for token presence (-2.0 to 2.0)
    reasoning_effort: str = "medium"  # Reasoning effort level (low, medium, high)
    safety_identifier: str = ""  # Safety identifier for content filtering
    service_tier: str = ""  # Service tier for API usage
    stop: List[str] = field(default_factory=list)  # Stop sequences
    temperature: float = DEFAULT_TEMPERATURE  # Controls randomness (0.0 to 2.0)
    top_p: float = 1.0  # Nucleus sampling parameter (0.0 to 1.0)
    max_tokens: int = None  # Maximum number of tokens to generate(deprecated, use max_completion_tokens instead)
    extra_body: dict = None  # Extra body parameters for OpenRouter of vLLM

    # Help descriptions
    _help: ClassVar[dict[str, str]] = {
        "frequency_penalty": "Penalty for token frequency (-2.0 to 2.0, positive values discourage repetition)",
        "max_completion_tokens": "Maximum number of tokens to generate (optional, leave empty for model default)",
        "presence_penalty": "Penalty for token presence (-2.0 to 2.0, positive values encourage new topics)",
        "reasoning_effort": "Reasoning effort level for o1 models (low, medium, high)",
        "safety_identifier": "Safety identifier for content filtering (optional)",
        "service_tier": "Service tier for API usage (optional)",
        "stop": 'Stop sequences (JSON array of strings, e.g., \'["</s>", "\\n\\n"]\')',
        "temperature": "Controls randomness (0.0-2.0, higher = more creative)",
        "top_p": "Nucleus sampling parameter (0.0-1.0, lower = more focused)",
        "max_tokens": "Maximum number of tokens to generate (deprecated, use max_completion_tokens instead)",
        "extra_body": 'Extra body parameters for OpenRouter of vLLM (JSON dict, e.g., \'"reasoning": {"reasoning": {"enabled": false}}\')',
    }


if __name__ == "__main__":
    import sys


    if len(sys.argv) > 1 and sys.argv[1] == "test":
        parser = ArgumentParser(description="Test binding options")
        OpenAILLMOptions.add_args(parser)

        # Parse arguments test
        args = parser.parse_args(
            [
                "--openai-llm-temperature",
                "0.7",
                "--openai-llm-max_completion_tokens",
                "1000",
                "--openai-llm-stop",
                '["</s>", "\\n\\n"]',
                "--openai-llm-reasoning",
                '{"effort": "high", "max_tokens": 2000, "exclude": false, "enabled": true}',
            ]
        )
        print("Final args for LLM and Embedding:")
        print(f"{args}\n")

        print("\nOpenAI LLM options:")
        print(OpenAILLMOptions.options_dict(args))

        # Test creating OpenAI options instance
        openai_options = OpenAILLMOptions(
            temperature=0.8,
            max_completion_tokens=1500,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stop=["<|end|>", "\n\n"],
        )
        print("\nOpenAI LLM options instance:")
        print(openai_options.asdict())

        # Test creating OpenAI options instance with reasoning parameter
        openai_options_with_reasoning = OpenAILLMOptions(
            temperature=0.9,
            max_completion_tokens=2000,
            reasoning={
                "effort": "medium",
                "max_tokens": 1500,
                "exclude": True,
                "enabled": True,
            },
        )
        print("\nOpenAI LLM options instance with reasoning:")
        print(openai_options_with_reasoning.asdict())

        # Test dict parsing functionality
        print("\n" + "=" * 50)
        print("TESTING DICT PARSING FUNCTIONALITY")
        print("=" * 50)

        # Test valid JSON dict parsing
        test_parser = ArgumentParser(description="Test dict parsing")
        OpenAILLMOptions.add_args(test_parser)

        try:
            test_args = test_parser.parse_args(
                ["--openai-llm-reasoning", '{"effort": "low", "max_tokens": 1000}']
            )
            print("✓ Valid JSON dict parsing successful:")
            print(
                f"  Parsed reasoning: {OpenAILLMOptions.options_dict(test_args)['reasoning']}"
            )
        except Exception as e:
            print(f"✗ Valid JSON dict parsing failed: {e}")

        # Test invalid JSON dict parsing
        try:
            test_args = test_parser.parse_args(
                [
                    "--openai-llm-reasoning",
                    '{"effort": "low", "max_tokens": 1000',  # Missing closing brace
                ]
            )
            print("✗ Invalid JSON should have failed but didn't")
        except SystemExit:
            print("✓ Invalid JSON dict parsing correctly rejected")
        except Exception as e:
            print(f"✓ Invalid JSON dict parsing correctly rejected: {e}")

        # Test non-dict JSON parsing
        try:
            test_args = test_parser.parse_args(
                [
                    "--openai-llm-reasoning",
                    '["not", "a", "dict"]',  # Array instead of dict
                ]
            )
            print("✗ Non-dict JSON should have failed but didn't")
        except SystemExit:
            print("✓ Non-dict JSON parsing correctly rejected")
        except Exception as e:
            print(f"✓ Non-dict JSON parsing correctly rejected: {e}")

        print("\n" + "=" * 50)
        print("TESTING ENVIRONMENT VARIABLE SUPPORT")
        print("=" * 50)

        # Test environment variable support for dict
        import os

        os.environ["OPENAI_LLM_REASONING"] = (
            '{"effort": "high", "max_tokens": 3000, "exclude": false}'
        )

        env_parser = ArgumentParser(description="Test env var dict parsing")
        OpenAILLMOptions.add_args(env_parser)

        try:
            env_args = env_parser.parse_args(
                []
            )  # No command line args, should use env var
            reasoning_from_env = OpenAILLMOptions.options_dict(env_args).get(
                "reasoning"
            )
            if reasoning_from_env:
                print("✓ Environment variable dict parsing successful:")
                print(f"  Parsed reasoning from env: {reasoning_from_env}")
            else:
                print("✗ Environment variable dict parsing failed: No reasoning found")
        except Exception as e:
            print(f"✗ Environment variable dict parsing failed: {e}")
        finally:
            # Clean up environment variable
            if "OPENAI_LLM_REASONING" in os.environ:
                del os.environ["OPENAI_LLM_REASONING"]

    else:
        print(BindingOptions.generate_dot_env_sample())
