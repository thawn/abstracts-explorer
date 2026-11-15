/**
 * API Utility Functions
 *
 * This module provides utility functions for API calls.
 */

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
            throw new Error(
                'The server took too long to respond. Please narrow the request ' +
                '(for example, fewer years or conferences) and try again.'
            );
        }
        throw new Error(`The server returned an invalid response (HTTP ${response.status}). Please try again.`);
    }

    if (response.ok === false) {
        throw new Error((data && data.error) || `HTTP error! status: ${response.status}`);
    }

    return data;
}
