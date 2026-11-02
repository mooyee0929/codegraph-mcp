"""Text helpers unrelated to payments."""


def slugify_title(title: str) -> str:
    """Turn an article title into a lowercase URL slug."""
    return "-".join(title.lower().split())


def parseHTTPHeaderLine(line: str) -> tuple[str, str]:
    """Split one raw header line into its name and value."""
    name, _, value = line.partition(":")
    return name.strip(), value.strip()
