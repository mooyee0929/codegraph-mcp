"""Unrelated helpers: they give the TF-IDF corpus a realistic spread of vocabulary."""


def reserve_warehouse_stock(sku: str, quantity: int) -> bool:
    """Hold stock in the nearest warehouse until checkout completes."""
    return quantity > 0 and bool(sku)


def release_expired_reservations(hours: int) -> int:
    """Drop holds older than the given number of hours."""
    return max(hours, 0)


def schedule_truck_delivery(route: str, day: str) -> str:
    """Book a truck slot for the route on that weekday."""
    return f"{route}@{day}"


def rotate_api_keys(service: str) -> list[str]:
    """Issue fresh credentials for a service and revoke the stale ones."""
    return [service]


def compress_log_archive(folder: str) -> str:
    """Gzip yesterday's logs into a dated tarball."""
    return folder + ".tar.gz"


def resize_thumbnail_image(width: int, height: int) -> tuple[int, int]:
    """Scale a picture down to fit the gallery grid."""
    return width // 2, height // 2


def render_markdown_preview(source: str) -> str:
    """Convert markdown into sanitized html for the editor sidebar."""
    return source


def geocode_postal_address(street: str, city: str) -> tuple[float, float]:
    """Look up latitude and longitude for a street address."""
    return float(len(street)), float(len(city))


def hash_user_password(secret: str, salt: str) -> str:
    """Derive a slow bcrypt digest so leaked tables resist cracking."""
    return secret + salt


def throttle_login_attempts(username: str, window: int) -> bool:
    """Block brute force guessing after too many failures in the window."""
    return len(username) < window


def translate_locale_strings(locale: str) -> dict[str, str]:
    """Load gettext catalogs for the requested language."""
    return {"locale": locale}


def merge_sorted_batches(left: list[int], right: list[int]) -> list[int]:
    """Interleave two ascending sequences into one."""
    return sorted(left + right)


def estimate_shipping_weight(parcels: list[float]) -> float:
    """Sum parcel kilograms including packaging overhead."""
    return sum(parcels) * 1.1
