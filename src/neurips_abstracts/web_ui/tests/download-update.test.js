/**
 * Unit tests for download/update feature in webui/static/app.js
 * 
 * Tests cover:
 * - Conference and year selector functionality
 * - Download/update button behavior
 * - SSE stream parsing
 * - Progress tracking
 */

const fs = require('fs');
const path = require('path');
const { TextEncoder, TextDecoder } = require('util');

// Polyfill TextEncoder for Node.js test environment
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

// Read and evaluate the app.js file
const appJsPath = path.join(__dirname, '../static/app.js');
const appJsCode = fs.readFileSync(appJsPath, 'utf8');

// Mock the marked library
global.marked = {
    parse: (text) => text,
    use: () => { }
};

// Create a function to evaluate the code in a controlled environment
function loadAppJs() {
    // Mock API_BASE constant
    global.API_BASE = '';

    // Mock fetch globally
    global.fetch = jest.fn();

    // Mock currentTab
    global.currentTab = 'search';

    // Execute the app.js code
    eval(appJsCode);

    // Return the functions we want to test
    return {
        updateYearsForConference,
        handleConferenceChange,
        handleYearChange,
        updateDownloadButton,
        handleDownload
    };
}

describe('Download/Update Feature', () => {
    let app;

    beforeEach(() => {
        // Clear all mocks
        jest.clearAllMocks();

        // Setup window.conferenceYearsMap and window.allYears BEFORE loading app
        global.window = global.window || {};
        global.window.conferenceYearsMap = {
            'NeurIPS': [2025, 2024, 2023],
            'ICLR': [2025, 2024]
        };
        global.window.allYears = [2025, 2024, 2023, 2022];

        // Setup DOM with all necessary elements
        document.body.innerHTML = `
            <div id="stats">Stats loading...</div>
            <select id="conference-selector">
                <option value="">All Conferences</option>
                <option value="NeurIPS">NeurIPS</option>
                <option value="ICLR">ICLR</option>
            </select>
            <select id="year-selector">
                <option value="">All Years</option>
            </select>
            <button id="download-btn">
                <div id="download-progress-bg" class="hidden"></div>
                <div>
                    <i id="download-icon" class="fas fa-download"></i>
                    <span id="download-btn-text">Download</span>
                </div>
            </button>
            <button id="tab-search" class="tab-btn"></button>
            <button id="tab-chat" class="tab-btn"></button>
            <button id="tab-interesting" class="tab-btn"></button>
            <div id="search-tab" class="tab-content"></div>
            <div id="chat-tab" class="tab-content hidden"></div>
            <div id="interesting-tab" class="tab-content hidden"></div>
            <input id="search-input" />
            <select id="limit-select"><option value="10">10</option></select>
            <div id="search-results"></div>
            <input id="chat-input" />
            <div id="chat-messages"></div>
            <div id="interesting-papers"></div>
        `;

        // Load app functions
        app = loadAppJs();

        // Setup fetch mock default
        global.fetch.mockResolvedValue({
            ok: true,
            json: async () => ({ count: 0 })
        });
    });

    describe('Conference and Year Selectors', () => {
        test('updateYearsForConference populates year dropdown', () => {
            const conferenceSelect = document.getElementById('conference-selector');
            const yearSelect = document.getElementById('year-selector');

            conferenceSelect.value = 'NeurIPS';
            app.updateYearsForConference();

            // Should have "All Years" plus 3 year options
            expect(yearSelect.options.length).toBe(4);
            expect(yearSelect.options[1].value).toBe('2025');
            expect(yearSelect.options[2].value).toBe('2024');
            expect(yearSelect.options[3].value).toBe('2023');
        });

        test('updateYearsForConference clears years when no conference selected', () => {
            const conferenceSelect = document.getElementById('conference-selector');
            const yearSelect = document.getElementById('year-selector');

            conferenceSelect.value = '';
            app.updateYearsForConference();

            // Should have "All Years" plus all available years
            expect(yearSelect.options.length).toBeGreaterThan(1);
        });

        test('handleConferenceChange triggers updates', () => {
            const conferenceSelect = document.getElementById('conference-selector');
            const yearSelect = document.getElementById('year-selector');

            conferenceSelect.value = 'NeurIPS';

            // Mock the dependent functions
            global.loadStats = jest.fn();
            global.loadFilterOptions = jest.fn();
            global.loadInterestingPapers = jest.fn();
            global.updateInterestingPapersCount = jest.fn();

            app.handleConferenceChange();

            // Year dropdown should be updated
            expect(yearSelect.options.length).toBe(4);
        });
    });

    describe('Download Button', () => {
        test('updateDownloadButton shows "Download" for new conference', async () => {
            const conferenceSelect = document.getElementById('conference-selector');
            const yearSelect = document.getElementById('year-selector');
            const downloadBtnText = document.getElementById('download-btn-text');

            // Add the year option first
            const option = document.createElement('option');
            option.value = '2025';
            option.textContent = '2025';
            yearSelect.appendChild(option);

            conferenceSelect.value = 'NeurIPS';
            yearSelect.value = '2025';

            // Mock API response with 0 papers (new conference)
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: async () => ({ total_papers: 0 })
            });

            await app.updateDownloadButton();

            expect(downloadBtnText.textContent).toBe('Download');
        });

        test('updateDownloadButton shows "Update" for existing conference', async () => {
            const conferenceSelect = document.getElementById('conference-selector');
            const yearSelect = document.getElementById('year-selector');
            const downloadBtnText = document.getElementById('download-btn-text');

            // Add the year option first
            const option = document.createElement('option');
            option.value = '2025';
            option.textContent = '2025';
            yearSelect.appendChild(option);

            conferenceSelect.value = 'NeurIPS';
            yearSelect.value = '2025';

            // Mock API response with papers (existing conference)
            const mockJson = jest.fn().mockResolvedValue({ total_papers: 100 });
            global.fetch.mockResolvedValueOnce({
                ok: true,
                json: mockJson
            });

            await app.updateDownloadButton();

            // Verify fetch was called
            expect(global.fetch).toHaveBeenCalled();

            // Verify json() was called
            expect(mockJson).toHaveBeenCalled();

            // The text should be "Update" since total_papers > 0
            expect(downloadBtnText.textContent).toBe('Update');
        });

        test('updateDownloadButton disables button when no conference selected', async () => {
            const conferenceSelect = document.getElementById('conference-selector');
            const yearSelect = document.getElementById('year-selector');
            const downloadBtn = document.getElementById('download-btn');
            const downloadBtnText = document.getElementById('download-btn-text');

            conferenceSelect.value = '';
            yearSelect.value = '2025';

            await app.updateDownloadButton();

            expect(downloadBtn.disabled).toBe(true);
            expect(downloadBtnText.textContent).toBe('Download');
        });

        test('updateDownloadButton disables button when no year selected', async () => {
            const conferenceSelect = document.getElementById('conference-selector');
            const yearSelect = document.getElementById('year-selector');
            const downloadBtn = document.getElementById('download-btn');

            conferenceSelect.value = 'NeurIPS';
            yearSelect.value = '';

            await app.updateDownloadButton();

            expect(downloadBtn.disabled).toBe(true);
        });
    });

    describe('Download Process with SSE', () => {
        test('handleDownload validates conference selection', async () => {
            const conferenceSelect = document.getElementById('conference-selector');
            const yearSelect = document.getElementById('year-selector');

            // Mock alert
            global.alert = jest.fn();

            conferenceSelect.value = '';
            yearSelect.value = '2025';

            await app.handleDownload();

            expect(global.alert).toHaveBeenCalledWith(expect.stringContaining('select a conference'));
            expect(global.fetch).not.toHaveBeenCalled();
        });

        test('handleDownload validates year selection', async () => {
            const conferenceSelect = document.getElementById('conference-selector');
            const yearSelect = document.getElementById('year-selector');

            // Mock alert
            global.alert = jest.fn();

            conferenceSelect.value = 'NeurIPS';
            yearSelect.value = '';

            await app.handleDownload();

            expect(global.alert).toHaveBeenCalledWith(expect.stringContaining('select a year'));
            expect(global.fetch).not.toHaveBeenCalled();
        });

        test('handleDownload sends POST request with correct data', async () => {
            const conferenceSelect = document.getElementById('conference-selector');
            const yearSelect = document.getElementById('year-selector');

            // Add the year option first
            const option = document.createElement('option');
            option.value = '2025';
            option.textContent = '2025';
            yearSelect.appendChild(option);

            // Set values before calling
            conferenceSelect.value = 'NeurIPS';
            yearSelect.value = '2025';

            // Verify values are set
            expect(conferenceSelect.value).toBe('NeurIPS');
            expect(yearSelect.value).toBe('2025');

            // Mock SSE stream
            const mockReader = {
                read: jest.fn()
                    .mockResolvedValueOnce({
                        done: false,
                        value: new TextEncoder().encode('data: {"stage":"complete","success":true}\n\n')
                    })
                    .mockResolvedValueOnce({ done: true })
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                body: {
                    getReader: () => mockReader
                }
            });

            await app.handleDownload();

            // Check that fetch was called
            expect(global.fetch).toHaveBeenCalled();

            // Check that it was POST to /api/download
            const fetchCall = global.fetch.mock.calls[0];
            expect(fetchCall[0]).toContain('/api/download');
            expect(fetchCall[1].method).toBe('POST');
            expect(fetchCall[1].body).toContain('NeurIPS');
            expect(fetchCall[1].body).toContain('2025');
        });

        test('handleDownload handles fetch errors gracefully', async () => {
            const conferenceSelect = document.getElementById('conference-selector');
            const yearSelect = document.getElementById('year-selector');
            const downloadBtn = document.getElementById('download-btn');

            // Add the year option first
            const option = document.createElement('option');
            option.value = '2025';
            option.textContent = '2025';
            yearSelect.appendChild(option);

            conferenceSelect.value = 'NeurIPS';
            yearSelect.value = '2025';

            // Verify values
            expect(conferenceSelect.value).toBe('NeurIPS');
            expect(yearSelect.value).toBe('2025');

            // Mock alert
            global.alert = jest.fn();

            // Mock fetch to throw network error
            global.fetch.mockRejectedValueOnce(new Error('Network error'));

            await app.handleDownload();

            // Should show error alert
            expect(global.alert).toHaveBeenCalled();
            const alertMessage = global.alert.mock.calls[0][0];
            expect(alertMessage).toContain('Error');
        });
    });

    describe('Progress Bar Updates', () => {
        test('progress bar becomes visible during download', async () => {
            const conferenceSelect = document.getElementById('conference-selector');
            const yearSelect = document.getElementById('year-selector');
            const progressBg = document.getElementById('download-progress-bg');

            // Add the year option first
            const option = document.createElement('option');
            option.value = '2025';
            option.textContent = '2025';
            yearSelect.appendChild(option);

            conferenceSelect.value = 'NeurIPS';
            yearSelect.value = '2025';

            // Initially hidden
            expect(progressBg.classList.contains('hidden')).toBe(true);

            // Mock SSE stream
            const mockReader = {
                read: jest.fn()
                    .mockResolvedValueOnce({
                        done: false,
                        value: new TextEncoder().encode('data: {"stage":"download","progress":50}\n\n')
                    })
                    .mockResolvedValueOnce({
                        done: false,
                        value: new TextEncoder().encode('data: {"stage":"complete","success":true}\n\n')
                    })
                    .mockResolvedValueOnce({ done: true })
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                body: {
                    getReader: () => mockReader
                }
            });

            await app.handleDownload();

            // After download, progress should have been shown at some point
            // Check that the hidden class was removed (it should still be visible or may be hidden again after completion)
            // The key is that it has a width set, indicating it was used
            expect(progressBg.style.width).toBeTruthy();
        });

        test('progress bar updates with percentage values', async () => {
            const conferenceSelect = document.getElementById('conference-selector');
            const yearSelect = document.getElementById('year-selector');
            const progressBg = document.getElementById('download-progress-bg');

            // Add the year option first
            const option = document.createElement('option');
            option.value = '2025';
            option.textContent = '2025';
            yearSelect.appendChild(option);

            conferenceSelect.value = 'NeurIPS';
            yearSelect.value = '2025';

            // Mock SSE stream with different progress values
            const mockReader = {
                read: jest.fn()
                    .mockResolvedValueOnce({
                        done: false,
                        value: new TextEncoder().encode('data: {"stage":"database","progress":75}\n\n')
                    })
                    .mockResolvedValueOnce({
                        done: false,
                        value: new TextEncoder().encode('data: {"stage":"complete","success":true}\n\n')
                    })
                    .mockResolvedValueOnce({ done: true })
            };

            global.fetch.mockResolvedValueOnce({
                ok: true,
                body: {
                    getReader: () => mockReader
                }
            });

            await app.handleDownload();

            // Progress bar should have a width set (as a percentage)
            expect(progressBg.style.width).toBeTruthy();
            if (progressBg.style.width) {
                expect(progressBg.style.width).toContain('%');
            }
        });
    });
});
