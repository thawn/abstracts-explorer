"""
ICLR Official Downloader Plugin
================================

Plugin for downloading papers from the official ICLR conference data.
"""

import logging

from neurips_abstracts.plugins.json_conference_downloader import JSONConferenceDownloaderPlugin

logger = logging.getLogger(__name__)


class ICLRDownloaderPlugin(JSONConferenceDownloaderPlugin):
    """
    Plugin for downloading papers from official ICLR conference.

    This plugin downloads data from the ICLR virtual conference site
    using their JSON API endpoint.
    """

    plugin_name = "iclr"
    plugin_description = "Official ICLR conference data downloader"
    supported_years = [2020, 2021, 2022, 2023, 2024, 2025]
    conference_name = "ICLR"

    def get_url(self, year: int) -> str:
        """
        Get the download URL for ICLR data.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        str
            URL to download ICLR JSON data
        """
        return f"https://iclr.cc/static/virtual/data/iclr-{year}-orals-posters.json"


# Auto-register the plugin when imported
def _register():
    """Auto-register the ICLR plugin."""
    from neurips_abstracts.plugins import register_plugin

    plugin = ICLRDownloaderPlugin()
    register_plugin(plugin)
    logger.debug("ICLR downloader plugin registered")


# Register on import
_register()
