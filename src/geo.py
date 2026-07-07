"""Geocoding — turns a user's free-text address into REAL coordinates.

The LLM extracts the address *string* only; it never emits coordinates (it would
hallucinate them, and a fabricated lat/lng looks authoritative). This module resolves
that string to ground-truth lat/lng + city via the Google Geocoding API, so the number
the model sees is real. It reuses the same Google Maps key used for the Places scrape.
"""
from __future__ import annotations

# Israel bounding box (lat_min, lat_max, lng_min, lng_max) — reject any geocode that
# lands outside it (a mis-geocode to the wrong country must never reach the model).
ISRAEL_BBOX = (29.4, 33.4, 34.2, 35.9)

# Google `location_type` precision. APPROXIMATE means the point is a city/region
# centroid, not a real street location — resolvable, but flag it as low confidence.
PRECISE_TYPES = {"ROOFTOP", "RANGE_INTERPOLATED", "GEOMETRIC_CENTER"}


def geocode_address(address: str, api_key: str) -> dict | None:
    """Resolve a free-text address to {lat, lng, city, precise, formatted}.

    Returns None on any failure (no key, network, no result, outside Israel) so the
    caller can fall back to the city path. Never raises."""
    if not address or not address.strip() or not api_key:
        return None
    try:
        import googlemaps

        client = googlemaps.Client(key=api_key)
        # region/language bias to Israel so the locality comes back in Hebrew, matching
        # the dataset's city names (e.g. "חיפה", not "Haifa").
        results = client.geocode(address.strip(), region="il", language="iw")
    except Exception:
        return None
    if not results:
        return None

    top = results[0]
    geometry = top.get("geometry", {})
    loc = geometry.get("location", {})
    lat, lng = loc.get("lat"), loc.get("lng")
    if lat is None or lng is None:
        return None

    lat_min, lat_max, lng_min, lng_max = ISRAEL_BBOX
    if not (lat_min <= lat <= lat_max and lng_min <= lng <= lng_max):
        return None  # geocoded outside Israel → refuse it rather than mislead the model

    # Read the real city (locality) off the geocoder's components — ground truth, not an
    # LLM guess, so it removes the need to trust an LLM-invented city entirely.
    city = None
    for comp in top.get("address_components", []):
        if "locality" in comp.get("types", []):
            city = comp.get("long_name")
            break

    return {
        "lat": float(lat),
        "lng": float(lng),
        "city": city,
        "precise": geometry.get("location_type") in PRECISE_TYPES,
        "formatted": top.get("formatted_address", address.strip()),
    }


def match_city(geo_city: str | None, valid_cities: list[str]) -> str | None:
    """Match a geocoded city name to a dataset city. Exact first, then a normalized
    contains-match (handles 'תל אביב-יפו' vs 'תל אביב'). Returns None if no confident
    match — the caller then ranks by distance from the real point instead of filtering
    to the wrong city, so a failed match only widens the pool, never misdirects it."""
    if not geo_city:
        return None
    if geo_city in valid_cities:
        return geo_city
    norm = geo_city.replace("-", " ").strip()
    for c in valid_cities:
        if c == norm or norm.startswith(c) or c.startswith(norm):
            return c
    return None
