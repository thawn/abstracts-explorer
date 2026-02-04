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
import { searchPapers } from './modules/search.js';
import { sendChatMessage, resetChat } from './modules/chat.js';
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
    openClusterSettings,
    closeClusterSettings,
    applyClusterSettings,
    exportClusters,
    toggleClusterParams,
    precalculateClusters
} from './modules/clustering.js';
import {
    loadFilterOptions,
    selectAllFilter,
    deselectAllFilter,
    openSearchSettings,
    openChatSettings,
    closeSettings,
    handleYearChange,
    handleConferenceChange
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

    // Close modal on Escape key
    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') {
            const modal = document.getElementById('settings-modal');
            if (modal && !modal.classList.contains('hidden')) {
                closeSettings();
            }
        }
    });
}

/**
 * Attach functions to window object for HTML event handlers
 * This is necessary because HTML onclick attributes need global functions
 */
function attachToWindow() {
    // Search module
    window.searchPapers = searchPapers;

    // Chat module
    window.sendChatMessage = sendChatMessage;
    window.resetChat = resetChat;

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

    // Clustering module
    window.loadClusters = loadClusters;
    window.openClusterSettings = openClusterSettings;
    window.closeClusterSettings = closeClusterSettings;
    window.applyClusterSettings = applyClusterSettings;
    window.exportClusters = exportClusters;
    window.toggleClusterParams = toggleClusterParams;
    window.precalculateClusters = precalculateClusters;

    // Filters module
    window.selectAllFilter = selectAllFilter;
    window.deselectAllFilter = deselectAllFilter;
    window.openSearchSettings = openSearchSettings;
    window.openChatSettings = openChatSettings;
    window.closeSettings = closeSettings;
    window.handleYearChange = handleYearChange;
    window.handleConferenceChange = handleConferenceChange;

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
