import re
import html
import logging
from .logger import logger

# Precompile regex pattern for JSON sanitization (module-level, compiled once)
_SURROGATE_PATTERN = re.compile(r"[\uD800-\uDFFF\uFFFE\uFFFF]")

# Global import for pypinyin with startup-time logging
try:
    import pypinyin
    _PYPINYIN_AVAILABLE = True
except ImportError:
    pypinyin = None
    _PYPINYIN_AVAILABLE = False
    logger.warning(
        "pypinyin is not installed. Chinese pinyin sorting will use simple string sorting."
    )

def _sanitize_string_for_json(text: str) -> str:
    """Remove characters that cannot be encoded in UTF-8 for JSON serialization.

    Uses regex for optimal performance with zero-copy optimization for clean strings.
    Fast detection path for clean strings (99% of cases) with efficient removal for dirty strings.

    Args:
        text: String to sanitize

    Returns:
        Original string if clean (zero-copy), sanitized string if dirty
    """
    if not text:
        return text

    # Fast path: Check if sanitization is needed using C-level regex search
    if not _SURROGATE_PATTERN.search(text):
        return text  # Zero-copy for clean strings - most common case

    # Slow path: Remove problematic characters using C-level regex substitution
    return _SURROGATE_PATTERN.sub("", text)


def safe_unicode_decode(content):
    # Regular expression to find all Unicode escape sequences of the form \uXXXX
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    # Function to replace the Unicode escape with the actual character
    def replace_unicode_escape(match):
        # Convert the matched hexadecimal value into the actual Unicode character
        return chr(int(match.group(1), 16))

    # Perform the substitution
    decoded_content = unicode_escape_pattern.sub(
        replace_unicode_escape, content.decode("utf-8")
    )

    return decoded_content


def sanitize_text_for_encoding(text: str, replacement_char: str = "") -> str:
    """Sanitize text to ensure safe UTF-8 encoding by removing or replacing problematic characters.

    This function handles:
    - Surrogate characters (the main cause of encoding errors)
    - Other invalid Unicode sequences
    - Control characters that might cause issues
    - Unescape HTML escapes
    - Remove control characters
    - Whitespace trimming

    Args:
        text: Input text to sanitize
        replacement_char: Character to use for replacing invalid sequences

    Returns:
        Sanitized text that can be safely encoded as UTF-8

    Raises:
        ValueError: When text contains uncleanable encoding issues that cannot be safely processed
    """
    if not text:
        return text

    try:
        # First, strip whitespace
        text = text.strip()

        # Early return if text is empty after basic cleaning
        if not text:
            return text

        # Try to encode/decode to catch any encoding issues early
        text.encode("utf-8")

        # Remove or replace surrogate characters (U+D800 to U+DFFF)
        # These are the main cause of the encoding error
        sanitized = ""
        for char in text:
            code_point = ord(char)
            # Check for surrogate characters
            if 0xD800 <= code_point <= 0xDFFF:
                # Replace surrogate with replacement character
                sanitized += replacement_char
                continue
            # Check for other problematic characters
            elif code_point == 0xFFFE or code_point == 0xFFFF:
                # These are non-characters in Unicode
                sanitized += replacement_char
                continue
            else:
                sanitized += char

        # Additional cleanup: remove null bytes and other control characters that might cause issues
        # (but preserve common whitespace like \t, \n, \r)
        sanitized = re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", replacement_char, sanitized
        )

        # Test final encoding to ensure it's safe
        sanitized.encode("utf-8")

        # Unescape HTML escapes
        sanitized = html.unescape(sanitized)

        # Remove control characters but preserve common whitespace (\t, \n, \r)
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", sanitized)

        return sanitized.strip()

    except UnicodeEncodeError as e:
        # Critical change: Don't return placeholder, raise exception for caller to handle
        error_msg = f"Text contains uncleanable UTF-8 encoding issues: {str(e)[:100]}"
        logger.error(f"Text sanitization failed: {error_msg}")
        raise ValueError(error_msg) from e

    except Exception as e:
        logger.error(f"Text sanitization: Unexpected error: {str(e)}")
        # For other exceptions, if no encoding issues detected, return original text
        try:
            text.encode("utf-8")
            return text
        except UnicodeEncodeError:
            raise ValueError(
                f"Text sanitization failed with unexpected error: {str(e)}"
            ) from e


def normalize_extracted_info(name: str, remove_inner_quotes=False) -> str:
    """Normalize entity/relation names and description with the following rules:
    - Clean HTML tags (paragraph and line break tags)
    - Convert Chinese symbols to English symbols
    - Remove spaces between Chinese characters
    - Remove spaces between Chinese characters and English letters/numbers
    - Preserve spaces within English text and numbers
    - Replace Chinese parentheses with English parentheses
    - Replace Chinese dash with English dash
    - Remove English quotation marks from the beginning and end of the text
    - Remove English quotation marks in and around chinese
    - Remove Chinese quotation marks
    - Filter out short numeric-only text (length < 3 and only digits/dots)
    - remove_inner_quotes = True
        remove Chinese quotes
        remove English quotes in and around chinese
        Convert non-breaking spaces to regular spaces
        Convert narrow non-breaking spaces after non-digits to regular spaces

    Args:
        name: Entity name to normalize
        is_entity: Whether this is an entity name (affects quote handling)

    Returns:
        Normalized entity name
    """
    # Clean HTML tags - remove paragraph and line break tags
    name = re.sub(r"</p\s*>|<p\s*>|<p/>", "", name, flags=re.IGNORECASE)
    name = re.sub(r"</br\s*>|<br\s*>|<br/>", "", name, flags=re.IGNORECASE)

    # Chinese full-width letters to half-width (A-Z, a-z)
    name = name.translate(
        str.maketrans(
            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )
    )

    # Chinese full-width numbers to half-width
    name = name.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

    # Chinese full-width symbols to half-width
    name = name.replace("－", "-")  # Chinese minus
    name = name.replace("＋", "+")  # Chinese plus
    name = name.replace("／", "/")  # Chinese slash
    name = name.replace("＊", "*")  # Chinese asterisk

    # Replace Chinese parentheses with English parentheses
    name = name.replace("（", "(").replace("）", ")")

    # Replace Chinese dash with English dash (additional patterns)
    name = name.replace("—", "-").replace("－", "-")

    # Chinese full-width space to regular space (after other replacements)
    name = name.replace("　", " ")

    # Use regex to remove spaces between Chinese characters
    # Regex explanation:
    # (?<=[\u4e00-\u9fa5]): Positive lookbehind for Chinese character
    # \s+: One or more whitespace characters
    # (?=[\u4e00-\u9fa5]): Positive lookahead for Chinese character
    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", name)

    # Remove spaces between Chinese and English/numbers/symbols
    name = re.sub(
        r"(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])", "", name
    )
    name = re.sub(
        r"(?<=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])\s+(?=[\u4e00-\u9fa5])", "", name
    )

    # Remove outer quotes
    if len(name) >= 2:
        # Handle double quotes
        if name.startswith('"') and name.endswith('"'):
            inner_content = name[1:-1]
            if '"' not in inner_content:  # No double quotes inside
                name = inner_content

        # Handle single quotes
        if name.startswith("'") and name.endswith("'"):
            inner_content = name[1:-1]
            if "'" not in inner_content:  # No single quotes inside
                name = inner_content

        # Handle Chinese-style double quotes
        if name.startswith("“") and name.endswith("”"):
            inner_content = name[1:-1]
            if "“" not in inner_content and "”" not in inner_content:
                name = inner_content
        if name.startswith("‘") and name.endswith("’"):
            inner_content = name[1:-1]
            if "‘" not in inner_content and "’" not in inner_content:
                name = inner_content

        # Handle Chinese-style book title mark
        if name.startswith("《") and name.endswith("》"):
            inner_content = name[1:-1]
            if "《" not in inner_content and "》" not in inner_content:
                name = inner_content

    if remove_inner_quotes:
        # Remove Chinese quotes
        name = name.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
        # Remove English queotes in and around chinese
        name = re.sub(r"['\"]+(?=[\u4e00-\u9fa5])", "", name)
        name = re.sub(r"(?<=[\u4e00-\u9fa5])['\"]+", "", name)
        # Convert non-breaking space to regular space
        name = name.replace("\u00a0", " ")
        # Convert narrow non-breaking space to regular space when after non-digits
        name = re.sub(r"(?<=[^\d])\u202F", " ", name)

    # Remove spaces from the beginning and end of the text
    name = name.strip()

    # Filter out pure numeric content with length < 3
    if len(name) < 3 and re.match(r"^[0-9]+$", name):
        return ""

    def should_filter_by_dots(text):
        """
        Check if the string consists only of dots and digits, with at least one dot
        Filter cases include: 1.2.3, 12.3, .123, 123., 12.3., .1.23 etc.
        """
        return all(c.isdigit() or c == "." for c in text) and "." in text

    if len(name) < 6 and should_filter_by_dots(name):
        # Filter out mixed numeric and dot content with length < 6
        return ""

    return name


def sanitize_and_normalize_extracted_text(
    input_text: str, remove_inner_quotes=False
) -> str:
    """Santitize and normalize extracted text
    Args:
        input_text: text string to be processed
        is_name: whether the input text is a entity or relation name

    Returns:
        Santitized and normalized text string
    """
    safe_input_text = sanitize_text_for_encoding(input_text)
    if safe_input_text:
        normalized_text = normalize_extracted_info(
            safe_input_text, remove_inner_quotes=remove_inner_quotes
        )
        return normalized_text
    return ""


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    content = content if content is not None else ""
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def is_float_regex(value: str) -> bool:
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def get_content_summary(content: str, max_length: int = 250) -> str:
    """Get summary of document content

    Args:
        content: Original document content
        max_length: Maximum length of summary

    Returns:
        Truncated content with ellipsis if needed
    """
    content = content.strip()
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def fix_tuple_delimiter_corruption(
    record: str, delimiter_core: str, tuple_delimiter: str
) -> str:
    """
    Fix various forms of tuple_delimiter corruption from LLM output.

    This function handles missing or replaced characters around the core delimiter.
    It fixes common corruption patterns where the LLM output doesn't match the expected
    tuple_delimiter format.

    Args:
        record: The text record to fix
        delimiter_core: The core delimiter (e.g., "S" from "<|#|>")
        tuple_delimiter: The complete tuple delimiter (e.g., "<|#|>")

    Returns:
        The corrected record with proper tuple_delimiter format
    """
    if not record or not delimiter_core or not tuple_delimiter:
        return record

    # Escape the delimiter core for regex use
    escaped_delimiter_core = re.escape(delimiter_core)

    # Fix: <|##|> -> <|#|>, <|#||#|> -> <|#|>, <|#|||#|> -> <|#|>
    record = re.sub(
        rf"<\|{escaped_delimiter_core}\|*?{escaped_delimiter_core}\|>",
        tuple_delimiter,
        record,
    )

    # Fix: <|\#|> -> <|#|>
    record = re.sub(
        rf"<\|\\{escaped_delimiter_core}\|>",
        tuple_delimiter,
        record,
    )

    # Fix: <|> -> <|#|>, <||> -> <|#|>
    record = re.sub(
        r"<\|+>",
        tuple_delimiter,
        record,
    )

    # Fix: <X|#|> -> <|#|>, <|#|Y> -> <|#|>, <X|#|Y> -> <|#|>, <||#||> -> <|#|> (one extra characters outside pipes)
    record = re.sub(
        rf"<.?\|{escaped_delimiter_core}\|.?>",
        tuple_delimiter,
        record,
    )

    # Fix: <#>, <#|>, <|#> -> <|#|> (missing one or both pipes)
    record = re.sub(
        rf"<\|?{escaped_delimiter_core}\|?>",
        tuple_delimiter,
        record,
    )

    # Fix: <X#|> -> <|#|>, <|#X> -> <|#|> (one pipe is replaced by other character)
    record = re.sub(
        rf"<[^|]{escaped_delimiter_core}\|>|<\|{escaped_delimiter_core}[^|]>",
        tuple_delimiter,
        record,
    )

    # Fix: <|#| -> <|#|>, <|#|| -> <|#|> (missing closing >)
    record = re.sub(
        rf"<\|{escaped_delimiter_core}\|+(?!>)",
        tuple_delimiter,
        record,
    )

    # Fix <|#: -> <|#|> (missing closing >)
    record = re.sub(
        rf"<\|{escaped_delimiter_core}:(?!>)",
        tuple_delimiter,
        record,
    )

    # Fix: <||#> -> <|#|> (double pipe at start, missing pipe at end)
    record = re.sub(
        rf"<\|+{escaped_delimiter_core}>",
        tuple_delimiter,
        record,
    )

    # Fix: <|| -> <|#|>
    record = re.sub(
        r"<\|\|(?!>)",
        tuple_delimiter,
        record,
    )

    # Fix: |#|> -> <|#|> (missing opening <)
    record = re.sub(
        rf"(?<!<)\|{escaped_delimiter_core}\|>",
        tuple_delimiter,
        record,
    )

    # Fix: <|#|>| -> <|#|>  ( this is a fix for: <|#|| -> <|#|> )
    record = re.sub(
        rf"<\|{escaped_delimiter_core}\|>\|",
        tuple_delimiter,
        record,
    )

    # Fix: ||#|| -> <|#|> (double pipes on both sides without angle brackets)
    record = re.sub(
        rf"\|\|{escaped_delimiter_core}\|\|",
        tuple_delimiter,
        record,
    )

    return record


def get_pinyin_sort_key(text: str) -> str:
    """Generate sort key for Chinese pinyin sorting

    This function uses pypinyin for true Chinese pinyin sorting.
    If pypinyin is not available, it falls back to simple lowercase string sorting.

    Args:
        text: Text to generate sort key for

    Returns:
        str: Sort key that can be used for comparison and sorting
    """
    if not text:
        return ""

    if _PYPINYIN_AVAILABLE:
        try:
            # Convert Chinese characters to pinyin, keep non-Chinese as-is
            pinyin_list = pypinyin.lazy_pinyin(text, style=pypinyin.Style.NORMAL)
            return "".join(pinyin_list).lower()
        except Exception:
            # Silently fall back to simple string sorting on any error
            return text.lower()
    else:
        # pypinyin not available, use simple string sorting
        return text.lower()
