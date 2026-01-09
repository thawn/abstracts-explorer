"""
ICML Official Downloader Plugin
================================

Plugin for downloading papers from the official ICML conference data.
"""

import logging

from abstracts_explorer.plugins.json_conference_downloader import JSONConferenceDownloaderPlugin

logger = logging.getLogger(__name__)


class ICMLDownloaderPlugin(JSONConferenceDownloaderPlugin):
    """
    Plugin for downloading papers from official ICML conference.

    This plugin downloads data from the ICML virtual conference site
    using their JSON API endpoint.
    """

    plugin_name = "icml"
    plugin_description = "Official ICML conference data downloader"
    supported_years = [2020, 2021, 2022, 2023, 2024, 2025]
    conference_name = "ICML"

    def get_url(self, year: int) -> str:
        """
        Get the download URL for ICML data.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        str
            URL to download ICML JSON data
        """
        return f"https://icml.cc/static/virtual/data/icml-{year}-orals-posters.json"


# Auto-register the plugin when imported
def _register():
    """Auto-register the ICML plugin."""
    from abstracts_explorer.plugins import register_plugin

    plugin = ICMLDownloaderPlugin()
    register_plugin(plugin)
    logger.debug("ICML downloader plugin registered")


# Register on import
_register()
