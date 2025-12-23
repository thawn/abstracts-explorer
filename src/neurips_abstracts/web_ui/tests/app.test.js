/**
 * Unit tests for webui/static/app.js
 * 
 * These tests cover the JavaScript functions in the NeurIPS Abstracts web UI.
 */

// Load the app.js file
const fs = require('fs');
const path = require('path');

// Read and evaluate the app.js file
const appJsPath = path.join(__dirname, '../static/app.js');
const appJsCode = fs.readFileSync(appJsPath, 'utf8');

// Mock the marked library for markdown parsing
global.marked = {
    parse: (text) => {
        // Simple mock that converts markdown to HTML
        let html = text;

        // Convert **bold** to <strong>
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Convert *italic* to <em>
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Convert # headers
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');

        // Convert lists
        html = html.replace(/^\* (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

        // Convert code blocks
        html = html.replace(/```(.+?)```/gs, '<pre><code>$1</code></pre>');
        html = html.replace(/`(.+?)`/g, '<code>$1</code>');

        // Convert paragraphs
        html = html.replace(/^(?!<[huplc])(.*?)$/gm, '<p>$1</p>');

        return html;
    }
};

// Create a function to evaluate the code in a controlled environment
function loadAppJs() {
    // Execute the app.js code
    eval(appJsCode);

    // Return the functions we want to test
    return {
        switchTab,
        loadStats,
        loadFilterOptions,
        selectAllFilter,
        deselectAllFilter,
        searchPapers,
        displaySearchResults,
        showPaperDetails,
        sendChatMessage,
        addChatMessage,
        resetChat,
        showError,
        escapeHtml,
        saveInterestingPapersAsJSON,
        loadInterestingPapersFromJSON,
        handleJSONFileLoad
    };
}

describe('NeurIPS Abstracts Web UI', () => {
    let app;

    beforeEach(() => {
        // Setup DOM
        document.body.innerHTML = `
            <div id="stats"></div>
            <select id="year-selector">
                <option value="">All Years</option>
            </select>
            <select id="conference-selector">
                <option value="">All Conferences</option>
            </select>
            <button id="tab-search" class="tab-btn border-purple-600 text-gray-700"></button>
            <button id="tab-chat" class="tab-btn border-transparent text-gray-500"></button>
            <button id="tab-interesting" class="tab-btn border-transparent text-gray-500"></button>
            <div id="search-tab" class="tab-content"></div>
            <div id="chat-tab" class="tab-content hidden"></div>
            <div id="interesting-tab" class="tab-content hidden"></div>
            <input id="search-input" value="" />
            <select id="limit-select"><option value="10">10</option></select>
            <select id="session-filter" multiple></select>
            <div id="search-results"></div>
            <input id="chat-input" value="" />
            <select id="n-papers"><option value="3">3</option></select>
            <select id="chat-session-filter" multiple></select>
            <div id="chat-messages"></div>
            <div id="chat-papers"></div>
            <div id="interesting-papers"></div>
        `;

        // Load app functions
        app = loadAppJs();
    });

    describe('escapeHtml', () => {
        test('should escape HTML special characters', () => {
            expect(app.escapeHtml('<script>alert("xss")</script>'))
                .toBe('&lt;script&gt;alert("xss")&lt;/script&gt;');
        });

        test('should escape ampersands', () => {
            expect(app.escapeHtml('Tom & Jerry')).toBe('Tom &amp; Jerry');
        });

        test('should handle quotes', () => {
            expect(app.escapeHtml('"Hello" \'World\'')).toContain('Hello');
        });

        test('should handle plain text', () => {
            expect(app.escapeHtml('Plain text')).toBe('Plain text');
        });

        test('should handle empty string', () => {
            expect(app.escapeHtml('')).toBe('');
        });
    });

    describe('switchTab', () => {
        test('should switch to search tab', () => {
            app.switchTab('search');

            const searchTab = document.getElementById('tab-search');
            const chatTab = document.getElementById('tab-chat');

            expect(searchTab.classList.contains('border-purple-600')).toBe(true);
            expect(searchTab.classList.contains('text-gray-700')).toBe(true);
            expect(chatTab.classList.contains('border-transparent')).toBe(true);
            expect(chatTab.classList.contains('text-gray-500')).toBe(true);
        });

        test('should switch to chat tab', () => {
            app.switchTab('chat');

            const searchTab = document.getElementById('tab-search');
            const chatTab = document.getElementById('tab-chat');

            expect(chatTab.classList.contains('border-purple-600')).toBe(true);
            expect(chatTab.classList.contains('text-gray-700')).toBe(true);
            expect(searchTab.classList.contains('border-transparent')).toBe(true);
        });

        test('should show correct tab content', () => {
            app.switchTab('chat');

            const searchContent = document.getElementById('search-tab');
            const chatContent = document.getElementById('chat-tab');

            expect(searchContent.classList.contains('hidden')).toBe(true);
            expect(chatContent.classList.contains('hidden')).toBe(false);
        });
    });

    describe('loadStats', () => {
        test('should load and display stats successfully', async () => {
            const mockStats = {
                total_papers: 12500,
                year: null,
                conference: null
            };

            fetch.mockResolvedValueOnce({
                json: async () => mockStats
            });

            await app.loadStats();

            const statsDiv = document.getElementById('stats');
            expect(statsDiv.innerHTML).toContain('12,500 Abstracts');
            expect(statsDiv.innerHTML).toContain('All Conferences');
        });

        test('should handle error response', async () => {
            fetch.mockResolvedValueOnce({
                json: async () => ({ error: 'Database not found' })
            });

            await app.loadStats();

            const statsDiv = document.getElementById('stats');
            expect(statsDiv.innerHTML).toContain('Database not found');
        });

        test('should handle fetch failure', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await app.loadStats();

            const statsDiv = document.getElementById('stats');
            expect(statsDiv.innerHTML).toContain('Error loading stats');
        });
    });

    describe('loadFilterOptions', () => {
        test('should load and populate filter options successfully', async () => {
            const mockFilters = {
                sessions: ['Session 1', 'Session 2']
            };

            const mockAvailableFilters = {
                conferences: ['NeurIPS', 'ICLR'],
                years: [2023, 2024, 2025],
                conference_years: {
                    'NeurIPS': [2023, 2024, 2025],
                    'ICLR': [2024, 2025]
                }
            };

            // Mock both fetch calls (filters first, then available-filters)
            fetch.mockResolvedValueOnce({
                json: async () => mockFilters
            });
            fetch.mockResolvedValueOnce({
                json: async () => mockAvailableFilters
            });

            await app.loadFilterOptions();

            const sessionSelect = document.getElementById('session-filter');

            expect(sessionSelect.options.length).toBe(2); // Two sessions

            // Check that sessions are selected by default
            expect(Array.from(sessionSelect.options).every(opt => opt.selected)).toBe(true);
        });

        test('should handle error response', async () => {
            fetch.mockResolvedValueOnce({
                json: async () => ({ error: 'Database error' })
            });

            await app.loadFilterOptions();

            // Should not throw, just log error
            const sessionSelect = document.getElementById('session-filter');
            expect(sessionSelect.options.length).toBe(0); // No options added
        });

        test('should handle fetch failure', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await app.loadFilterOptions();

            // Should not throw, just log error
            const sessionSelect = document.getElementById('session-filter');
            expect(sessionSelect.options.length).toBe(0); // No options added
        });
    });

    describe('selectAllFilter', () => {
        test('should select all options in the filter', () => {
            const sessionSelect = document.getElementById('session-filter');

            // Add some options
            const option1 = document.createElement('option');
            option1.value = 'Session 1';
            option1.textContent = 'Session 1';
            option1.selected = false;
            sessionSelect.appendChild(option1);

            const option2 = document.createElement('option');
            option2.value = 'Session 2';
            option2.textContent = 'Session 2';
            option2.selected = false;
            sessionSelect.appendChild(option2);

            // Load the app functions in the global scope to use selectAllFilter
            const appJs = loadAppJs();

            // Call selectAllFilter
            appJs.selectAllFilter('session-filter');

            // Check that all options are selected
            expect(option1.selected).toBe(true);
            expect(option2.selected).toBe(true);
        });
    });

    describe('deselectAllFilter', () => {
        test('should deselect all options in the filter', () => {
            const sessionSelect = document.getElementById('session-filter');

            // Add some options
            const option1 = document.createElement('option');
            option1.value = 'Session 1';
            option1.textContent = 'Session 1';
            option1.selected = true;
            sessionSelect.appendChild(option1);

            const option2 = document.createElement('option');
            option2.value = 'Session 2';
            option2.textContent = 'Session 2';
            option2.selected = true;
            sessionSelect.appendChild(option2);

            // Load the app functions in the global scope to use deselectAllFilter
            const appJs = loadAppJs();

            // Call deselectAllFilter
            appJs.deselectAllFilter('session-filter');

            // Check that all options are deselected
            expect(option1.selected).toBe(false);
            expect(option2.selected).toBe(false);
        });
    });

    describe('showError', () => {
        test('should display error message', () => {
            app.showError('Test error message');

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('Error');
            expect(resultsDiv.innerHTML).toContain('Test error message');
            expect(resultsDiv.innerHTML).toContain('fa-exclamation-circle');
        });

        test('should escape HTML in error message', () => {
            app.showError('<script>alert("xss")</script>');

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).not.toContain('<script>');
            expect(resultsDiv.innerHTML).toContain('&lt;script&gt;');
        });
    });

    describe('displaySearchResults', () => {
        test('should display no results message when empty', () => {
            const data = { papers: [], count: 0, use_embeddings: false };

            app.displaySearchResults(data);

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('No papers found');
            expect(resultsDiv.innerHTML).toContain('Try different keywords');
        });

        test('should display search results with papers', () => {
            const data = {
                papers: [
                    {
                        uid: 'test_uid_abc123',
                        title: 'Test Paper 1',
                        authors: ['Author A', 'Author B'],
                        abstract: 'This is a test abstract.',
                        session: 'Session 1A',
                        poster_position: '42',
                        paper_url: 'https://example.com/paper1'
                    },
                    {
                        uid: 'test_uid_def456',
                        title: 'Test Paper 2',
                        authors: ['Author C'],
                        abstract: 'Another test abstract.'
                    }
                ],
                count: 2,
                use_embeddings: false
            };

            app.displaySearchResults(data);

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('Found <strong>2</strong> papers');
            expect(resultsDiv.innerHTML).toContain('Test Paper 1');
            expect(resultsDiv.innerHTML).toContain('Test Paper 2');
            expect(resultsDiv.innerHTML).toContain('Author A, Author B');
            expect(resultsDiv.innerHTML).toContain('Author C');
            expect(resultsDiv.innerHTML).toContain('Session 1A');
            expect(resultsDiv.innerHTML).toContain('Poster 42');
            expect(resultsDiv.innerHTML).toContain('View Paper Details');
        });

        test('should show AI-Powered badge for embedding search', () => {
            const data = {
                papers: [{ uid: 'test_uid_123', title: 'Test', authors: [], abstract: 'Test' }],
                count: 1,
                use_embeddings: true
            };

            app.displaySearchResults(data);

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('AI-Powered');
        });

        test('should handle missing authors', () => {
            const data = {
                papers: [
                    {
                        uid: 'test_uid_789',
                        title: 'Test Paper',
                        authors: null,
                        abstract: 'Test abstract'
                    }
                ],
                count: 1,
                use_embeddings: false
            };

            app.displaySearchResults(data);

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('Unknown');
        });

        test('should display error for string authors', () => {
            const data = {
                papers: [
                    {
                        uid: 'test_uid_abc',
                        title: 'Test Paper',
                        authors: 'John Doe, Jane Smith', // String instead of array
                        abstract: 'Test abstract'
                    }
                ],
                count: 1,
                use_embeddings: false
            };

            app.displaySearchResults(data);

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('Error');
            expect(resultsDiv.innerHTML).toContain('Expected authors to be an array');
            expect(console.error).toHaveBeenCalled();
        });

        test('should display error for non-array authors', () => {
            const data = {
                papers: [
                    {
                        uid: 'test_uid_def',
                        title: 'Test Paper',
                        authors: { name: 'John Doe' }, // Object instead of array
                        abstract: 'Test abstract'
                    }
                ],
                count: 1,
                use_embeddings: false
            };

            app.displaySearchResults(data);

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('Error');
            expect(resultsDiv.innerHTML).toContain('Expected authors to be an array');
            expect(console.error).toHaveBeenCalled();
        });

        test('should use details element for long abstracts', () => {
            const longAbstract = 'A'.repeat(400);
            const data = {
                papers: [
                    {
                        uid: 'test_uid_ghi',
                        name: 'Test Paper',
                        authors: ['Author'],
                        abstract: longAbstract
                    }
                ],
                count: 1,
                use_embeddings: false
            };

            app.displaySearchResults(data);

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('<details');
            expect(resultsDiv.innerHTML).toContain('Show more');
            expect(resultsDiv.innerHTML).toContain('...');
            // Abstract is wrapped in markdown <p> tags, so check for presence of many A's
            expect(resultsDiv.innerHTML).toContain('AAAAAAAAAA');
        });

        test('should not use details element for short abstracts', () => {
            const shortAbstract = 'This is a short abstract.';
            const data = {
                papers: [
                    {
                        uid: 'test_uid_jkl',
                        name: 'Test Paper',
                        authors: ['Author'],
                        abstract: shortAbstract
                    }
                ],
                count: 1,
                use_embeddings: false
            };

            app.displaySearchResults(data);

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).not.toContain('<details');
            expect(resultsDiv.innerHTML).not.toContain('Show more');
            expect(resultsDiv.innerHTML).toContain(shortAbstract);
        });

        test('should display relevance score for embeddings', () => {
            const data = {
                papers: [
                    {
                        uid: 'test_uid_mno',
                        name: 'Test Paper',
                        authors: ['Author'],
                        abstract: 'Test',
                        distance: 0.25
                    }
                ],
                count: 1,
                use_embeddings: true
            };

            app.displaySearchResults(data);

            const resultsDiv = document.getElementById('search-results');
            // The score is now displayed in a badge with fa-chart-line icon
            expect(resultsDiv.innerHTML).toContain('fa-chart-line');
            expect(resultsDiv.innerHTML).toContain('0.750');
        });

        test('should escape HTML in paper data', () => {
            const data = {
                papers: [
                    {
                        uid: 'test_uid_pqr',
                        title: '<script>alert("xss")</script>',
                        authors: ['<b>Author</b>'],
                        abstract: '<img src=x onerror=alert(1)>'
                    }
                ],
                count: 1,
                use_embeddings: false
            };

            app.displaySearchResults(data);

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).not.toContain('<script>');
            // Title and authors should be escaped
            expect(resultsDiv.innerHTML).toContain('&lt;script&gt;');
            expect(resultsDiv.innerHTML).toContain('&lt;b&gt;Author&lt;/b&gt;');
            // Abstract is passed through markdown parser which allows some HTML like images
            expect(resultsDiv.innerHTML).toContain('<img');
        });

        test('should properly quote string UIDs in onclick handlers', () => {
            // Regression test: ensure onclick handlers have quoted UIDs for valid JavaScript
            const data = {
                papers: [
                    {
                        uid: 'test_uid_abc123',
                        name: 'Test Paper',
                        authors: ['Author'],
                        abstract: 'Test abstract'
                    }
                ],
                count: 1,
                use_embeddings: false
            };

            app.displaySearchResults(data);

            const resultsDiv = document.getElementById('search-results');

            // Check that onclick has properly quoted UID
            expect(resultsDiv.innerHTML).toContain("showPaperDetails('test_uid_abc123')");

            // Should NOT contain unquoted version (which would be invalid JavaScript)
            expect(resultsDiv.innerHTML).not.toContain('showPaperDetails(test_uid_abc123)');

            // Check star rating onclick also has proper quotes
            expect(resultsDiv.innerHTML).toContain("setPaperPriority('test_uid_abc123',");
        });

        test('should handle string UIDs with special characters in onclick', () => {
            // Test that UIDs with underscores, hyphens, and alphanumeric chars work
            const data = {
                papers: [
                    {
                        uid: 'uid_123-abc_xyz',
                        name: 'Test Paper',
                        authors: ['Author'],
                        abstract: 'Test'
                    }
                ],
                count: 1,
                use_embeddings: false
            };

            app.displaySearchResults(data);

            const resultsDiv = document.getElementById('search-results');

            // Verify proper quoting
            expect(resultsDiv.innerHTML).toContain("showPaperDetails('uid_123-abc_xyz')");
            expect(resultsDiv.innerHTML).toContain("setPaperPriority('uid_123-abc_xyz',");
        });
    });

    describe('searchPapers', () => {
        test('should show error when query is empty', async () => {
            document.getElementById('search-input').value = '';

            await app.searchPapers();

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('Please enter a search query');
        });

        test('should send correct request data', async () => {
            const searchInput = document.getElementById('search-input');
            const limitSelect = document.getElementById('limit-select');

            searchInput.value = 'neural networks';

            // Create a new option and add it
            const option = document.createElement('option');
            option.value = '20';
            option.selected = true;
            limitSelect.appendChild(option);

            fetch.mockResolvedValueOnce({
                json: async () => ({ papers: [], count: 0 })
            });

            await app.searchPapers();

            expect(fetch).toHaveBeenCalledWith(
                '/api/search',
                expect.objectContaining({
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: expect.stringContaining('neural networks')
                })
            );

            const callBody = JSON.parse(fetch.mock.calls[0][1].body);
            expect(callBody.query).toBe('neural networks');
            expect(callBody.use_embeddings).toBe(true);
        });

        test('should send filter values when selected', async () => {
            const searchInput = document.getElementById('search-input');
            const sessionSelect = document.getElementById('session-filter');

            searchInput.value = 'machine learning';

            // Add filter options - some selected, some not
            // This ensures filters are sent (app only sends filters when NOT all are selected)
            const sessionOption1 = document.createElement('option');
            sessionOption1.value = 'Session 1';
            sessionOption1.selected = true;
            sessionSelect.appendChild(sessionOption1);

            const sessionOption2 = document.createElement('option');
            sessionOption2.value = 'Session 2';
            sessionOption2.selected = true;
            sessionSelect.appendChild(sessionOption2);

            const sessionOption3 = document.createElement('option');
            sessionOption3.value = 'Session 3';
            sessionOption3.selected = false; // Not selected, so filters will be sent
            sessionSelect.appendChild(sessionOption3);

            fetch.mockResolvedValueOnce({
                json: async () => ({ papers: [], count: 0 })
            });

            await app.searchPapers();

            const callBody = JSON.parse(fetch.mock.calls[0][1].body);
            expect(callBody.query).toBe('machine learning');
            expect(callBody.sessions).toEqual(['Session 1', 'Session 2']);
        });

        test('should handle API error response', async () => {
            document.getElementById('search-input').value = 'test';

            fetch.mockResolvedValueOnce({
                json: async () => ({ error: 'Database error' })
            });

            await app.searchPapers();

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('Database error');
        });

        test('should handle fetch failure', async () => {
            document.getElementById('search-input').value = 'test';

            fetch.mockRejectedValueOnce(new Error('Network error'));

            await app.searchPapers();

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('error occurred while searching');
        });
    });

    describe('addChatMessage', () => {
        test('should add user message', () => {
            const messageId = app.addChatMessage('Hello', 'user');

            const messagesDiv = document.getElementById('chat-messages');
            expect(messagesDiv.innerHTML).toContain('Hello');
            expect(messagesDiv.innerHTML).toContain('fa-user');
            expect(messagesDiv.innerHTML).toContain('bg-purple-600');
            expect(messageId).toContain('msg-');
        });

        test('should add assistant message', () => {
            const messageId = app.addChatMessage('Hi there!', 'assistant');

            const messagesDiv = document.getElementById('chat-messages');
            expect(messagesDiv.innerHTML).toContain('Hi there!');
            expect(messagesDiv.innerHTML).toContain('fa-robot');
            expect(messageId).toContain('msg-');
        });

        test('should add loading indicator', () => {
            app.addChatMessage('Loading...', 'assistant', true);

            const messagesDiv = document.getElementById('chat-messages');
            expect(messagesDiv.innerHTML).toContain('spinner');
        });

        test('should escape HTML in user messages', () => {
            app.addChatMessage('<script>alert("xss")</script>', 'user');

            const messagesDiv = document.getElementById('chat-messages');
            expect(messagesDiv.innerHTML).not.toContain('<script>');
            expect(messagesDiv.innerHTML).toContain('&lt;script&gt;');
        });

        test('should render markdown in assistant messages', () => {
            app.addChatMessage('**Bold text** and *italic text*', 'assistant');

            const messagesDiv = document.getElementById('chat-messages');
            expect(messagesDiv.innerHTML).toContain('<strong>Bold text</strong>');
            expect(messagesDiv.innerHTML).toContain('<em>italic text</em>');
            expect(messagesDiv.innerHTML).toContain('markdown-content');
        });

        test('should render markdown headers in assistant messages', () => {
            app.addChatMessage('# Heading 1\n## Heading 2', 'assistant');

            const messagesDiv = document.getElementById('chat-messages');
            expect(messagesDiv.innerHTML).toContain('<h1>Heading 1</h1>');
            expect(messagesDiv.innerHTML).toContain('<h2>Heading 2</h2>');
        });

        test('should render markdown code in assistant messages', () => {
            app.addChatMessage('Use `code` for inline code', 'assistant');

            const messagesDiv = document.getElementById('chat-messages');
            expect(messagesDiv.innerHTML).toContain('<code>code</code>');
        });

        test('should render markdown lists in assistant messages', () => {
            app.addChatMessage('* Item 1\n* Item 2', 'assistant');

            const messagesDiv = document.getElementById('chat-messages');
            expect(messagesDiv.innerHTML).toContain('<li>Item 1</li>');
            expect(messagesDiv.innerHTML).toContain('<li>Item 2</li>');
        });
    });

    describe('sendChatMessage', () => {
        test('should not send empty message', async () => {
            document.getElementById('chat-input').value = '';

            await app.sendChatMessage();

            expect(fetch).not.toHaveBeenCalled();
        });

        test('should send message and display response', async () => {
            const chatInput = document.getElementById('chat-input');
            const nPapersSelect = document.getElementById('n-papers');

            chatInput.value = 'What is deep learning?';

            // Create a new option and add it
            const option = document.createElement('option');
            option.value = '5';
            option.selected = true;
            nPapersSelect.appendChild(option);

            fetch.mockResolvedValueOnce({
                json: async () => ({
                    response: 'Deep learning is a subset of machine learning...'
                })
            });

            await app.sendChatMessage();

            expect(fetch).toHaveBeenCalledWith(
                '/api/chat',
                expect.objectContaining({
                    method: 'POST',
                    body: expect.stringContaining('What is deep learning?')
                })
            );

            const messagesDiv = document.getElementById('chat-messages');
            // User message gets replaced by assistant response, so just check for response
            expect(messagesDiv.innerHTML).toContain('Deep learning is a subset of machine learning');
        });

        test('should handle response as dictionary object', async () => {
            const chatInput = document.getElementById('chat-input');
            const nPapersSelect = document.getElementById('n-papers');

            chatInput.value = 'Test query';

            const option = document.createElement('option');
            option.value = '3';
            option.selected = true;
            nPapersSelect.appendChild(option);

            // Backend returns response as a dictionary with nested response field
            fetch.mockResolvedValueOnce({
                json: async () => ({
                    response: {
                        response: 'This is the actual response text',
                        papers: [],
                        metadata: {}
                    }
                })
            });

            await app.sendChatMessage();

            const messagesDiv = document.getElementById('chat-messages');
            expect(messagesDiv.innerHTML).toContain('This is the actual response text');
        });

        test('should clear input after sending', async () => {
            const input = document.getElementById('chat-input');
            input.value = 'Test message';

            fetch.mockResolvedValueOnce({
                json: async () => ({ response: 'Response' })
            });

            await app.sendChatMessage();

            expect(input.value).toBe('');
        });

        test('should handle error response', async () => {
            document.getElementById('chat-input').value = 'Test';

            fetch.mockResolvedValueOnce({
                json: async () => ({ error: 'LM Studio not available' })
            });

            await app.sendChatMessage();

            const messagesDiv = document.getElementById('chat-messages');
            expect(messagesDiv.innerHTML).toContain('Error: LM Studio not available');
        });

        test('should handle fetch failure', async () => {
            document.getElementById('chat-input').value = 'Test';

            fetch.mockRejectedValueOnce(new Error('Network error'));

            await app.sendChatMessage();

            const messagesDiv = document.getElementById('chat-messages');
            expect(messagesDiv.innerHTML).toContain('error occurred');
        });
    });

    describe('resetChat', () => {
        test('should reset chat conversation', async () => {
            // Add some messages first
            app.addChatMessage('Hello', 'user');
            app.addChatMessage('Hi!', 'assistant');

            fetch.mockResolvedValueOnce({ ok: true });

            await app.resetChat();

            expect(fetch).toHaveBeenCalledWith(
                '/api/chat/reset',
                expect.objectContaining({ method: 'POST' })
            );

            const messagesDiv = document.getElementById('chat-messages');
            expect(messagesDiv.innerHTML).toContain('Conversation reset');
            expect(messagesDiv.innerHTML).not.toContain('Hello');
        });

        test('should handle reset failure', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await app.resetChat();

            // Should not throw error
            expect(console.error).toHaveBeenCalled();
        });
    });

    describe('showPaperDetails', () => {
        test('should fetch and display paper details', async () => {
            const mockPaper = {
                uid: 'test_uid_stu',
                title: 'Test Paper',
                authors: ['Author A', 'Author B'],
                abstract: 'This is a test abstract',
                session: 'Session 2B',
                poster_position: '101',
                url: 'https://example.com/paper',
                pdf_url: 'https://example.com/paper.pdf',
                paper_url: 'https://example.com/paper_link'
            };

            fetch.mockResolvedValueOnce({
                json: async () => mockPaper
            });

            await app.showPaperDetails('test_uid_stu');

            expect(fetch).toHaveBeenCalledWith('/api/paper/test_uid_stu');

            // Check if modal was created
            const modal = document.querySelector('.fixed');
            expect(modal).toBeTruthy();
            expect(modal.innerHTML).toContain('Test Paper');
            expect(modal.innerHTML).toContain('Author A, Author B');
            expect(modal.innerHTML).toContain('This is a test abstract');
            expect(modal.innerHTML).toContain('Session 2B');
            expect(modal.innerHTML).toContain('Poster 101');
            expect(modal.innerHTML).toContain('View Paper Details');
            expect(modal.innerHTML).toContain('View PDF');
            expect(modal.innerHTML).toContain('Paper Link');
        });

        test('should handle missing PDF URL', async () => {
            const mockPaper = {
                uid: 'test_uid_vwx',
                name: 'Test Paper',
                authors: [],
                abstract: 'Test'
            };

            fetch.mockResolvedValueOnce({
                json: async () => mockPaper
            });

            await app.showPaperDetails('test_uid_vwx');

            const modal = document.querySelector('.fixed');
            expect(modal.innerHTML).not.toContain('View PDF');
        });

        test('should handle error response', async () => {
            fetch.mockResolvedValueOnce({
                json: async () => ({ error: 'Paper not found' })
            });

            await app.showPaperDetails(999);

            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('Paper not found');
        });

        test('should escape HTML in paper details', async () => {
            const mockPaper = {
                uid: 'test_uid_yza',
                title: '<script>alert("xss")</script>',
                authors: ['<b>Author</b>'],
                abstract: '<img src=x onerror=alert(1)>'
            };

            fetch.mockResolvedValueOnce({
                json: async () => mockPaper
            });

            await app.showPaperDetails('test_uid_yza');

            const modal = document.querySelector('.fixed');
            expect(modal.innerHTML).not.toContain('<script>');
            // Title and authors should be escaped
            expect(modal.innerHTML).toContain('&lt;script&gt;');
            expect(modal.innerHTML).toContain('&lt;b&gt;Author&lt;/b&gt;');
            // Abstract is passed through markdown parser which allows some HTML like images
            expect(modal.innerHTML).toContain('<img');
        });

        test('should handle string authors error in paper details', async () => {
            const mockPaper = {
                id: 1,
                name: 'Test Paper',
                authors: 'John Doe, Jane Smith', // String instead of array
                abstract: 'Test abstract'
            };

            fetch.mockResolvedValueOnce({
                json: async () => mockPaper
            });

            await app.showPaperDetails('test_uid_bcd');

            // Error should be displayed
            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('Error');
            expect(console.error).toHaveBeenCalled();
        });

        test('should handle non-array authors error in paper details', async () => {
            const mockPaper = {
                uid: 'test_uid_efg',
                name: 'Test Paper',
                authors: 123, // Number instead of array
                abstract: 'Test abstract'
            };

            fetch.mockResolvedValueOnce({
                json: async () => mockPaper
            });

            await app.showPaperDetails('test_uid_efg');

            // Error should be displayed
            const resultsDiv = document.getElementById('search-results');
            expect(resultsDiv.innerHTML).toContain('Error');
            expect(console.error).toHaveBeenCalled();
        });
    });

    describe('JSON Import/Export', () => {
        let mockCreateElement;
        let mockBlob;
        let mockURL;
        let originalAlert;
        let originalConfirm;
        let testApp;

        beforeEach(() => {
            // Mock alert and confirm
            originalAlert = global.alert;
            originalConfirm = global.confirm;
            global.alert = jest.fn();
            global.confirm = jest.fn(() => true);

            // Mock URL.createObjectURL and revokeObjectURL
            mockURL = {
                createObjectURL: jest.fn(() => 'blob:mock-url'),
                revokeObjectURL: jest.fn()
            };
            global.URL = mockURL;

            // Mock Blob
            mockBlob = jest.fn((content, options) => ({
                content,
                options
            }));
            global.Blob = mockBlob;

            // Store original createElement
            mockCreateElement = document.createElement;

            // Mock localStorage
            global.localStorage = {
                getItem: jest.fn(),
                setItem: jest.fn(),
                clear: jest.fn()
            };

            // Setup sort order select element
            const sortSelect = document.createElement('select');
            sortSelect.id = 'sort-order';
            document.body.appendChild(sortSelect);

            // Create mocks for functions that may be called - make them available globally
            global.updateInterestingPapersCount = jest.fn();
            global.loadInterestingPapers = jest.fn();
            global.savePriorities = jest.fn();

            // Reload the app with the mocked functions available
            testApp = loadAppJs();
        });

        afterEach(() => {
            global.alert = originalAlert;
            global.confirm = originalConfirm;
        });

        describe('saveInterestingPapersAsJSON', () => {
            test('should alert if no papers are rated', () => {
                // Call with empty paperPriorities (default state)
                testApp.saveInterestingPapersAsJSON();

                expect(global.alert).toHaveBeenCalledWith(
                    'No papers rated yet. Rate some papers before saving.'
                );
            });

            // Note: Full integration tests for save/load are performed manually
            // as the test framework has limitations accessing closure variables
        });

        describe('loadInterestingPapersFromJSON', () => {
            test('should trigger file input click', () => {
                const mockFileInput = {
                    click: jest.fn()
                };
                const originalGetElementById = document.getElementById;
                document.getElementById = jest.fn(() => mockFileInput);

                testApp.loadInterestingPapersFromJSON();

                expect(mockFileInput.click).toHaveBeenCalled();

                document.getElementById = originalGetElementById;
            });
        });

        describe('handleJSONFileLoad', () => {
            let mockEvent;
            let mockFileReader;

            beforeEach(() => {
                mockFileReader = {
                    onload: null,
                    onerror: null,
                    readAsText: jest.fn(function () {
                        // Simulate async file read
                        setTimeout(() => {
                            if (this.onload) {
                                this.onload({ target: { result: this._mockResult } });
                            }
                        }, 0);
                    }),
                    _mockResult: ''
                };

                global.FileReader = jest.fn(() => mockFileReader);

                mockEvent = {
                    target: {
                        files: [],
                        value: ''
                    }
                };
            });

            test('should return early if no file selected', () => {
                mockEvent.target.files = [];

                testApp.handleJSONFileLoad(mockEvent);

                expect(global.FileReader).not.toHaveBeenCalled();
            });

            test('should reject non-JSON files', () => {
                mockEvent.target.files = [{ name: 'test.txt' }];

                testApp.handleJSONFileLoad(mockEvent);

                expect(global.alert).toHaveBeenCalledWith('Please select a JSON file.');
                expect(mockEvent.target.value).toBe('');
            });

            // Note: The following tests verify JSON parsing and validation logic
            // Full integration tests with localStorage are performed manually
            // as the test framework has limitations accessing closure variables

            test('should handle invalid JSON format', async () => {
                mockFileReader._mockResult = 'invalid json{';
                mockEvent.target.files = [{ name: 'test.json' }];

                testApp.handleJSONFileLoad(mockEvent);

                // Wait for FileReader to process
                await new Promise(resolve => setTimeout(resolve, 15));

                expect(global.alert).toHaveBeenCalledWith(
                    expect.stringContaining('Error loading JSON file')
                );
            });

            test('should handle missing paperPriorities field', async () => {
                mockFileReader._mockResult = JSON.stringify({
                    version: '1.0',
                    exportDate: '2025-12-14T09:00:00.000Z'
                    // Missing paperPriorities
                });
                mockEvent.target.files = [{ name: 'test.json' }];

                testApp.handleJSONFileLoad(mockEvent);

                // Wait for FileReader to process
                await new Promise(resolve => setTimeout(resolve, 15));

                expect(global.alert).toHaveBeenCalledWith(
                    expect.stringContaining('Invalid JSON format: missing or invalid paperPriorities')
                );
            });

            test('should handle file read error', async () => {
                mockEvent.target.files = [{ name: 'test.json' }];

                global.FileReader = jest.fn(function () {
                    return {
                        onload: null,
                        onerror: null,
                        readAsText: jest.fn(function () {
                            setTimeout(() => {
                                if (this.onerror) {
                                    this.onerror();
                                }
                            }, 0);
                        })
                    };
                });

                testApp.handleJSONFileLoad(mockEvent);

                // Wait for FileReader to process
                await new Promise(resolve => setTimeout(resolve, 15));

                expect(global.alert).toHaveBeenCalledWith('Error reading file. Please try again.');
            });
        });
    });
});
