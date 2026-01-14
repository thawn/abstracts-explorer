/**
 * API Utility Functions
 * 
 * This module provides utility functions for API calls and request building.
 */

import { API_BASE } from './constants.js';

/**
 * Get selected filters from DOM
 * @returns {Object} Object with yearSelect, conferenceSelect, selectedYear, selectedConference
 */
export function getSelectedFilters() {
    const yearSelect = document.getElementById('year-selector');
    const conferenceSelect = document.getElementById('conference-selector');
    return {
        yearSelect: yearSelect,
        conferenceSelect: conferenceSelect,
        selectedYear: yearSelect ? yearSelect.value : '',
        selectedConference: conferenceSelect ? conferenceSelect.value : ''
    };
}

/**
 * Build filtered request body
 * @param {Object} baseRequest - Base request object
 * @param {Array} sessions - Selected sessions
 * @param {number} totalSessions - Total number of session options
 * @param {string} selectedYear - Selected year
 * @param {string} selectedConference - Selected conference
 * @returns {Object} Request body with filters
 */
export function buildFilteredRequestBody(baseRequest, sessions, totalSessions, selectedYear, selectedConference) {
    const requestBody = { ...baseRequest };
    
    // Add filters only if NOT all options are selected (all selected = no filter)
    if (sessions.length > 0 && sessions.length < totalSessions) {
        requestBody.sessions = sessions;
    }
    
    // Add year filter if selected
    if (selectedYear) {
        requestBody.years = [parseInt(selectedYear)];
    }
    
    // Add conference filter if selected
    if (selectedConference) {
        requestBody.conferences = [selectedConference];
    }
    
    return requestBody;
}

/**
 * Apply year and conference filters to papers
 * @param {Array} papers - Array of papers to filter
 * @param {string} selectedYear - Year to filter by
 * @param {string} selectedConference - Conference to filter by
 * @returns {Array} Filtered papers
 */
export function applyYearConferenceFilters(papers, selectedYear, selectedConference) {
    let filteredPapers = papers;
    
    if (selectedYear) {
        filteredPapers = filteredPapers.filter(paper => 
            String(paper.year) === String(selectedYear)
        );
    }
    
    if (selectedConference) {
        filteredPapers = filteredPapers.filter(paper => 
            paper.conference === selectedConference
        );
    }
    
    return filteredPapers;
}

/**
 * Fetch JSON from API with error handling
 * @param {string} url - URL to fetch
 * @param {Object} options - Fetch options
 * @returns {Promise<Object>} Response data
 */
export async function fetchJSON(url, options = {}) {
    const response = await fetch(`${API_BASE}${url}`, options);
    const data = await response.json();
    
    if (data.error) {
        throw new Error(data.error);
    }
    
    return data;
}
