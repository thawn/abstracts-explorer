"""
NeurIPS Official Downloader Plugin
===================================

Plugin for downloading papers from the official NeurIPS conference data.
"""

import logging

from abstracts_explorer.plugins.json_conference_downloader import JSONConferenceDownloaderPlugin

logger = logging.getLogger(__name__)


class NeurIPSDownloaderPlugin(JSONConferenceDownloaderPlugin):
    """
    Plugin for downloading papers from official NeurIPS conference.

    This plugin downloads data from the NeurIPS virtual conference site
    using their JSON API endpoint.
    """

    plugin_name = "neurips"
    plugin_description = "Official NeurIPS conference data downloader"
    supported_years = list(range(2020, 2026))  # NeurIPS years available
    conference_name = "NeurIPS"

    def get_url(self, year: int) -> str:
        """
        Get the download URL for NeurIPS data.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        str
            URL to download NeurIPS JSON data
        """
        return f"https://neurips.cc/static/virtual/data/neurips-{year}-orals-posters.json"


# Auto-register the plugin when imported
def _register():
    """Auto-register the NeurIPS plugin."""
    from abstracts_explorer.plugins import register_plugin

    plugin = NeurIPSDownloaderPlugin()
    register_plugin(plugin)
    logger.debug("NeurIPS downloader plugin registered")


# Register on import
_register()
