/**
 * Filter Utility Functions
 * 
 * This module provides utility functions for working with filters.
 */

/**
 * Get selected year and conference from header selectors
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
 * Apply year and conference filters to papers
 * @param {Array} papers - Array of papers to filter
 * @param {string} selectedYear - Year to filter by (empty string for no filter)
 * @param {string} selectedConference - Conference to filter by (empty string for no filter)
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
 * Build request body with filters
 * @param {Object} baseRequest - Base request object
 * @param {Array} sessions - Selected sessions (empty array or partial selection only)
 * @param {number} totalSessions - Total number of session options
 * @param {string} selectedYear - Selected year filter
 * @param {string} selectedConference - Selected conference filter
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
