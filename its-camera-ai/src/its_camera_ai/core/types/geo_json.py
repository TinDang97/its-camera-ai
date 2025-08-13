from typing import Any, TypedDict


class Geometry(TypedDict):
    """GeoJSON Geometry representation."""

    type: str
    coordinates: list[float]


class GeoJSONFeature(TypedDict):
    """GeoJSON Feature representation."""

    type: str
    geometry: Geometry
    properties: dict[str, Any]
