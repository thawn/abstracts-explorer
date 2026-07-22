"""
AISTATS Official Downloader Plugin
===================================

Plugin for downloading papers from the official AISTATS conference data
via the EventHosts JSON API.
"""

import logging

from abstracts_explorer.plugins.json_conference_downloader import JSONConferenceDownloaderPlugin

logger = logging.getLogger(__name__)


class AISTATSDownloaderPlugin(JSONConferenceDownloaderPlugin):
    """
    Plugin for downloading papers from the official AISTATS conference.

    This plugin downloads data from the AISTATS virtual conference site
    using the EventHosts JSON API endpoint.
    """

    plugin_name = "aistats"
    plugin_description = "Official AISTATS conference data downloader"
    _start_year = 2021
    conference_name = "AISTATS"

    def get_url(self, year: int) -> str:
        """
        Get the download URL for AISTATS data.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        str
            URL to download AISTATS JSON data
        """
        return f"https://virtual.aistats.org/static/virtual/data/aistats-{year}-orals-posters.json"


# Auto-register the plugin when imported
def _register():
    """Auto-register the AISTATS plugin."""
    from abstracts_explorer.plugins import register_plugin

    plugin = AISTATSDownloaderPlugin()
    register_plugin(plugin)
    logger.debug("AISTATS downloader plugin registered")


# Register on import
_register()
