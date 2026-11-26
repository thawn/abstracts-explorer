/**
 * Main Application Entry Point
 *
 * This module initializes the application, imports all feature modules,
 * and attaches functions to the window object for HTML event handlers.
 */

import { configureMarkedWithKatex } from './modules/utils/markdown-utils.js';

import { loadPriorities } from './modules/state.js';

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
    switchInterestingConferenceFromMenu,
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
    filterSessionOptions,
    clearSessionSearch,
    openSearchSettings,
    openChatSettings,
    closeSettings,
    handleYearChange,
    handleConferenceChange,
    handleLocationChange,
    dismissConferenceError,
    dismissDefaultFilterError
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
 * Initialize the application.
 */
function initializeApp(): void {
    configureMarkedWithKatex();
    loadPriorities();

    loadStats();
    loadFilterOptions();
    checkEmbeddingModelCompatibility();

    updateInterestingPapersCount();

    initMcpToolsHint();

    setupModalEventListeners();
}

/**
 * Toggle mobile navigation menu.
 */
function toggleMobileMenu(): void {
    const controls = document.getElementById('header-controls');
    const btn = document.getElementById('mobile-menu-btn');
    if (!controls) return;

    const isHidden = controls.classList.contains('hidden');
    if (isHidden) {
        controls.classList.remove('hidden');
        controls.classList.add('flex');
        if (btn) {
            btn.setAttribute('aria-expanded', 'true');
            const icon = btn.querySelector('i');
            if (icon) {
                icon.classList.remove('fa-bars');
                icon.classList.add('fa-times');
            }
        }
    } else {
        controls.classList.remove('flex');
        controls.classList.add('hidden');
        if (btn) {
            btn.setAttribute('aria-expanded', 'false');
            const icon = btn.querySelector('i');
            if (icon) {
                icon.classList.remove('fa-times');
                icon.classList.add('fa-bars');
            }
        }
    }
}

/**
 * Setup modal event listeners.
 */
function setupModalEventListeners(): void {
    const settingsModal = document.getElementById('settings-modal');
    if (settingsModal) {
        settingsModal.addEventListener('click', function (event) {
            if (event.target === settingsModal) {
                closeSettings();
            }
        });
    }

    const advancedSearchModal = document.getElementById('advanced-search-modal');
    if (advancedSearchModal) {
        advancedSearchModal.addEventListener('click', function (event) {
            if (event.target === advancedSearchModal) {
                closeAdvancedSearch();
            }
        });
    }

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

    const dtInput = document.getElementById('distance-threshold-input') as HTMLInputElement | null;
    if (dtInput) {
        dtInput.addEventListener('input', function () {
            localStorage.setItem('searchDistanceThreshold', dtInput.value);
        });
    }
}

/**
 * Attach functions to window object for HTML event handlers.
 */
function attachToWindow(): void {
    window.searchPapers = searchPapers;
    window.openAdvancedSearch = openAdvancedSearch;
    window.closeAdvancedSearch = closeAdvancedSearch;
    window.applyAdvancedSearch = applyAdvancedSearch;

    window.sendChatMessage = sendChatMessage;
    window.resetChat = resetChat;
    window.openPapersModal = openPapersModal;
    window.closePapersModal = closePapersModal;
    window.handleChatFeedback = handleChatFeedback;

    window.loadInterestingPapers = loadInterestingPapers;
    window.saveInterestingPapersAsMarkdown = saveInterestingPapersAsMarkdown;
    window.saveInterestingPapersAsJSON = saveInterestingPapersAsJSON;
    window.loadInterestingPapersFromJSON = loadInterestingPapersFromJSON;
    window.handleJSONFileLoad = handleJSONFileLoad;
    window.editSearchTerm = editSearchTerm;
    window.editPaperSearchTerm = editPaperSearchTerm;
    window.switchInterestingSession = switchInterestingSession;
    window.changeInterestingPapersSortOrder = changeInterestingPapersSortOrder;
    window.switchInterestingConferenceFromMenu = switchInterestingConferenceFromMenu;
    window.donateInterestingPapersData = donateInterestingPapersData;
    window.updateControlsVisibility = updateControlsVisibility;
    window.loadPriorities = loadPriorities;

    window.loadClusters = loadClusters;
    window.exportClusters = exportClusters;
    window.resetClusters = resetClusters;
    window.loadPapersPerYear = loadPapersPerYear;

    window.selectAllFilter = selectAllFilter;
    window.deselectAllFilter = deselectAllFilter;
    window.filterSessionOptions = filterSessionOptions;
    window.clearSessionSearch = clearSessionSearch;
    window.openSearchSettings = openSearchSettings;
    window.openChatSettings = openChatSettings;
    window.closeSettings = closeSettings;
    window.handleYearChange = handleYearChange;
    window.handleConferenceChange = handleConferenceChange;
    window.handleLocationChange = handleLocationChange;
    window.dismissConferenceError = dismissConferenceError;
    window.dismissDefaultFilterError = dismissDefaultFilterError;

    window.switchTab = switchTab;
    window.loadStats = loadStats;
    window.dismissEmbeddingWarning = dismissEmbeddingWarning;

    window.showPaperDetails = showPaperDetails;
    window.setPaperPriority = setPaperPriority;
    window.updateInterestingPapersCount = updateInterestingPapersCount;

    window.toggleMobileMenu = toggleMobileMenu;
}

document.addEventListener('DOMContentLoaded', function () {
    attachToWindow();
    initializeApp();
});
