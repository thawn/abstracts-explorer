/**
 * Ambient declarations for globals injected by the Flask templates and for
 * third-party libraries that ship without TypeScript types.
 */

interface UrlConferenceError {
    message?: string;
    available_conferences?: string[];
}

interface DefaultFilterError {
    message?: string;
    available_conferences?: string[];
    available_years?: Array<string | number>;
    guidance?: string;
}

interface Window {
    urlConference?: string;
    urlConferenceError?: UrlConferenceError;
    conferenceYearsMap?: Record<string, Array<string | number>>;
    allYears?: Array<string | number>;

    searchPapers?: (...args: any[]) => any;
    openAdvancedSearch?: (...args: any[]) => any;
    closeAdvancedSearch?: (...args: any[]) => any;
    applyAdvancedSearch?: (...args: any[]) => any;

    sendChatMessage?: (...args: any[]) => any;
    resetChat?: (...args: any[]) => any;
    openPapersModal?: (...args: any[]) => any;
    closePapersModal?: (...args: any[]) => any;
    handleChatFeedback?: (...args: any[]) => any;

    loadInterestingPapers?: (...args: any[]) => any;
    saveInterestingPapersAsMarkdown?: (...args: any[]) => any;
    saveInterestingPapersAsJSON?: (...args: any[]) => any;
    loadInterestingPapersFromJSON?: (...args: any[]) => any;
    handleJSONFileLoad?: (...args: any[]) => any;
    editSearchTerm?: (...args: any[]) => any;
    editPaperSearchTerm?: (...args: any[]) => any;
    switchInterestingSession?: (...args: any[]) => any;
    changeInterestingPapersSortOrder?: (...args: any[]) => any;
    donateInterestingPapersData?: (...args: any[]) => any;
    updateControlsVisibility?: (...args: any[]) => any;
    loadPriorities?: (...args: any[]) => any;

    loadClusters?: (...args: any[]) => any;
    exportClusters?: (...args: any[]) => any;
    resetClusters?: (...args: any[]) => any;
    loadPapersPerYear?: (...args: any[]) => any;

    selectAllFilter?: (...args: any[]) => any;
    deselectAllFilter?: (...args: any[]) => any;
    openSearchSettings?: (...args: any[]) => any;
    openChatSettings?: (...args: any[]) => any;
    closeSettings?: (...args: any[]) => any;
    handleYearChange?: (...args: any[]) => any;
    handleConferenceChange?: (...args: any[]) => any;
    dismissConferenceError?: (...args: any[]) => any;
    dismissDefaultFilterError?: (...args: any[]) => any;

    switchTab?: (...args: any[]) => any;
    loadStats?: (...args: any[]) => any;
    dismissEmbeddingWarning?: (...args: any[]) => any;

    showPaperDetails?: (...args: any[]) => any;
    setPaperPriority?: (...args: any[]) => any;
    updateInterestingPapersCount?: (...args: any[]) => any;

    toggleMobileMenu?: (...args: any[]) => any;

    event?: Event;
}

declare module "plotly.js-dist-min" {
    const Plotly: any;
    export default Plotly;
}
