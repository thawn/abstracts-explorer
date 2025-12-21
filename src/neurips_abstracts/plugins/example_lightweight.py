"""
Example Lightweight Plugin
===========================

This is an example plugin demonstrating the lightweight API for simple data sources.
"""

from typing import Any, Dict, List, Optional
import logging

from neurips_abstracts.plugin import (
    LightweightDownloaderPlugin,
    convert_lightweight_to_neurips_schema,
    register_plugin,
)

logger = logging.getLogger(__name__)


class ExampleLightweightPlugin(LightweightDownloaderPlugin):
    """
    Example plugin demonstrating the lightweight API.

    This plugin shows how to create a simple downloader that only needs
    to provide the essential fields, without worrying about the full
    NeurIPS schema complexity.
    """

    plugin_name = "example_lightweight"
    plugin_description = "Example lightweight plugin for demonstration"
    supported_years = [2024, 2025]

    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Download papers using lightweight format.

        Parameters
        ----------
        year : int, optional
            Year to download (default: 2025)
        output_path : str, optional
            Path to save data
        force_download : bool
            Force re-download
        **kwargs : Any
            Additional parameters

        Returns
        -------
        dict
            Data in full NeurIPS schema format
        """
        if year is None:
            year = 2025

        self.validate_year(year)

        logger.info(f"Downloading example papers for {year}...")

        # Example papers in lightweight format
        # Only required fields: title, authors, abstract, session, poster_position
        papers = [
            {
                "id": 1,
                "title": "Example Paper: Machine Learning for Physical Sciences",
                "authors": ["Alice Johnson", "Bob Smith", "Carol Davis"],
                "abstract": (
                    "This is an example paper demonstrating the lightweight plugin API. "
                    "The abstract should contain a summary of the research. "
                    "This plugin automatically converts simple data to the full NeurIPS schema."
                ),
                "session": "Machine Learning Applications",
                "poster_position": "A1",
                # Optional fields below
                "paper_pdf_url": "https://example.com/papers/ml_physics_2025.pdf",
                "url": "https://example.com/papers/1",
                "poster_image_url": "https://example.com/posters/1.png",
                "room_name": "Hall A",
                "keywords": ["machine learning", "physics", "simulation"],
                "starttime": "2025-12-10T09:00:00",
                "endtime": "2025-12-10T10:30:00",
            },
            {
                "id": 2,
                "title": "Deep Learning for Climate Modeling",
                "authors": [
                    {"fullname": "David Lee", "institution": "MIT"},
                    {"fullname": "Emma Wilson", "institution": "Stanford"},
                ],
                "abstract": (
                    "We present a novel approach to climate modeling using deep learning. "
                    "Our method achieves state-of-the-art results on benchmark datasets. "
                    "Note: authors can be provided as dicts with institution info."
                ),
                "session": "Climate and Environment",
                "poster_position": "B3",
                "paper_pdf_url": "https://example.com/papers/climate_dl_2025.pdf",
                "url": "https://openreview.net/forum?id=example123",
                "keywords": ["deep learning", "climate", "forecasting"],
            },
            {
                "id": 3,
                "title": "Minimal Example: Only Required Fields",
                "authors": ["Frank Miller"],
                "abstract": "This paper demonstrates minimal required fields only.",
                "session": "Poster Session",
                "poster_position": "C5",
            },
        ]

        # Convert lightweight format to full NeurIPS schema
        data = convert_lightweight_to_neurips_schema(
            papers,
            session_default="Example Workshop 2025",
            event_type="Workshop Poster",
            source_url="https://example-workshop.com/2025",
        )

        logger.info(f"Successfully converted {data['count']} papers to full schema")

        return data

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.

        Returns
        -------
        dict
            Plugin metadata
        """
        return {
            "name": self.plugin_name,
            "description": self.plugin_description,
            "supported_years": self.supported_years,
            "api_type": "lightweight",
            "required_fields": ["title", "authors", "abstract", "session", "poster_position"],
            "optional_fields": [
                "id",
                "paper_pdf_url",
                "poster_image_url",
                "url",
                "room_name",
                "keywords",
                "starttime",
                "endtime",
            ],
            "parameters": {
                "year": {"type": "int", "required": True, "description": "Workshop year", "default": 2025},
                "output_path": {"type": "str", "required": False, "description": "Path to save the downloaded data"},
                "force_download": {
                    "type": "bool",
                    "required": False,
                    "description": "Force re-download",
                    "default": False,
                },
            },
        }


# Auto-register the plugin when imported (commented out for example)
# def _register():
#     """Auto-register the example plugin."""
#     plugin = ExampleLightweightPlugin()
#     register_plugin(plugin)
#     logger.debug("Example lightweight plugin registered")
#
# _register()
