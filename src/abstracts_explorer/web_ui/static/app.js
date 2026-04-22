/**
 * Main Application Entry Point
 * 
 * This module initializes the application, imports all feature modules,
 * and attaches functions to the window object for HTML event handlers.
 */

// Import utility functions
import { configureMarkedWithKatex } from './modules/utils/markdown-utils.js';

// Import state management
import { loadPriorities } from './modules/state.js';

// Import feature modules
import { searchPapers, openAdvancedSearch, closeAdvancedSearch, applyAdvancedSearch } from './modules/search.js';
import { sendChatMessage, resetChat, openPapersModal, closePapersModal, handleChatFeedback, initMcpToolsHint } from './modules/chat.js';
import {
    loadInterestingPapers,
    saveInterestingPapersAsMarkdown,
    saveInterestingPapersAsJSON,
    loadInterestingPapersFromJSON,
    handleJSONFileLoad,
    editSearchTerm,
    editPaperSearchTerm,
    switchInterestingSession,
    changeInterestingPapersSortOrder,
    donateInterestingPapersData,
    updateControlsVisibility
} from './modules/interesting-papers.js';
import {
    loadClusters,
    exportClusters,
    resetClusters,
    loadPapersPerYear
} from './modules/clustering.js';
import {
    loadFilterOptions,
    selectAllFilter,
    deselectAllFilter,
    openSearchSettings,
    openChatSettings,
    closeSettings,
    handleYearChange,
    handleConferenceChange,
    dismissConferenceError
} from './modules/filters.js';
import {
    switchTab,
    loadStats,
    checkEmbeddingModelCompatibility,
    dismissEmbeddingWarning
} from './modules/tabs.js';
import {
    showPaperDetails,
    setPaperPriority,
    updateInterestingPapersCount
} from './modules/paper-card.js';

/**
 * Initialize the application
 */
function initializeApp() {
    // Configure markdown renderer with KaTeX
    configureMarkedWithKatex();

    // Load application state
    loadPriorities();

    // Load initial data
    loadStats();
    loadFilterOptions();
    checkEmbeddingModelCompatibility();

    // Update interesting papers count
    updateInterestingPapersCount();

    // Show MCP tools hint in chat area
    initMcpToolsHint();

    // Setup modal event listeners
    setupModalEventListeners();
}

/**
 * Setup modal event listeners
 */
function setupModalEventListeners() {
    // Close modal when clicking outside of it
    const settingsModal = document.getElementById('settings-modal');
    if (settingsModal) {
        settingsModal.addEventListener('click', function (event) {
            if (event.target === this) {
                closeSettings();
            }
        });
    }

    const advancedSearchModal = document.getElementById('advanced-search-modal');
    if (advancedSearchModal) {
        advancedSearchModal.addEventListener('click', function (event) {
            if (event.target === this) {
                closeAdvancedSearch();
            }
        });
    }

    // Close modal on Escape key
    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') {
            const modal = document.getElementById('settings-modal');
            if (modal && !modal.classList.contains('hidden')) {
                closeSettings();
            }
            const advModal = document.getElementById('advanced-search-modal');
            if (advModal && !advModal.classList.contains('hidden')) {
                closeAdvancedSearch();
            }
        }
    });

    // Allow Enter key to submit advanced search
    const advancedSearchFields = ['adv-topic', 'adv-authors', 'adv-title', 'adv-keywords', 'adv-abstract', 'adv-award'];
    advancedSearchFields.forEach(id => {
        const input = document.getElementById(id);
        if (input) {
            input.addEventListener('keypress', function (event) {
                if (event.key === 'Enter') {
                    applyAdvancedSearch();
                }
            });
        }
    });

    // Persist distance threshold to localStorage on every value change (including while typing)
    const dtInput = document.getElementById('distance-threshold-input');
    if (dtInput) {
        dtInput.addEventListener('input', function () {
            localStorage.setItem('searchDistanceThreshold', this.value);
        });
    }
}

/**
 * Attach functions to window object for HTML event handlers
 * This is necessary because HTML onclick attributes need global functions
 */
function attachToWindow() {
    // Search module
    window.searchPapers = searchPapers;
    window.openAdvancedSearch = openAdvancedSearch;
    window.closeAdvancedSearch = closeAdvancedSearch;
    window.applyAdvancedSearch = applyAdvancedSearch;

    // Chat module
    window.sendChatMessage = sendChatMessage;
    window.resetChat = resetChat;
    window.openPapersModal = openPapersModal;
    window.closePapersModal = closePapersModal;
    window.handleChatFeedback = handleChatFeedback;

    // Interesting papers module
    window.loadInterestingPapers = loadInterestingPapers;
    window.saveInterestingPapersAsMarkdown = saveInterestingPapersAsMarkdown;
    window.saveInterestingPapersAsJSON = saveInterestingPapersAsJSON;
    window.loadInterestingPapersFromJSON = loadInterestingPapersFromJSON;
    window.handleJSONFileLoad = handleJSONFileLoad;
    window.editSearchTerm = editSearchTerm;
    window.editPaperSearchTerm = editPaperSearchTerm;
    window.switchInterestingSession = switchInterestingSession;
    window.changeInterestingPapersSortOrder = changeInterestingPapersSortOrder;
    window.donateInterestingPapersData = donateInterestingPapersData;
    window.updateControlsVisibility = updateControlsVisibility;
    window.loadPriorities = loadPriorities;  // Expose for testing

    // Clustering module
    window.loadClusters = loadClusters;
    window.exportClusters = exportClusters;
    window.resetClusters = resetClusters;
    window.loadPapersPerYear = loadPapersPerYear;

    // Filters module
    window.selectAllFilter = selectAllFilter;
    window.deselectAllFilter = deselectAllFilter;
    window.openSearchSettings = openSearchSettings;
    window.openChatSettings = openChatSettings;
    window.closeSettings = closeSettings;
    window.handleYearChange = handleYearChange;
    window.handleConferenceChange = handleConferenceChange;
    window.dismissConferenceError = dismissConferenceError;

    // Tabs module
    window.switchTab = switchTab;
    window.loadStats = loadStats;
    window.dismissEmbeddingWarning = dismissEmbeddingWarning;

    // Paper card module
    window.showPaperDetails = showPaperDetails;
    window.setPaperPriority = setPaperPriority;
    window.updateInterestingPapersCount = updateInterestingPapersCount;
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function () {
    attachToWindow();
    initializeApp();
});
