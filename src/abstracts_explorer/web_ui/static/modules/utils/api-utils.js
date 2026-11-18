/**
 * API Utility Functions
 *
 * This module provides utility functions for API calls.
 */

const TIMEOUT_MESSAGE =
    'The server took too long to respond. Please try again. ' +
    'If this keeps happening, try a narrower request or fewer papers.';
const UNAVAILABLE_MESSAGE =
    'The server is temporarily unavailable. Please try again  later.';

/**
 * Parse a fetch response as JSON with helpful handling for non-JSON bodies.
 *
 * Reverse proxies may return an HTML error page (for example a 504 gateway
 * timeout) instead of the JSON produced by the application.  Calling
 * ``response.json()`` directly then throws a cryptic ``SyntaxError``.  This
 * helper converts that case into an actionable error message.
 *
 * @param {Response} response - Fetch response
 * @returns {Promise<Object>} Parsed JSON body
 */
export async function parseJsonResponse(response) {
    let data;
    try {
        data = await response.json();
    } catch (error) {
        if (response.status === 504 || response.status === 502) {
            throw new Error(TIMEOUT_MESSAGE);
        }
        if (response.status === 503) {
            throw new Error(UNAVAILABLE_MESSAGE);
        }
        throw new Error('The server returned an unexpected response. Please try again.');
    }

    if (response.ok === false) {
        if (data && data.error) {
            throw new Error(data.error);
        }
        if (response.status === 504 || response.status === 502) {
            throw new Error(TIMEOUT_MESSAGE);
        }
        if (response.status === 503) {
            throw new Error(UNAVAILABLE_MESSAGE);
        }
        throw new Error(`The request could not be completed (HTTP ${response.status}). Please try again.`);
    }

    return data;
}
