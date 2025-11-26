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

// Create a function to evaluate the code in a controlled environment
function loadAppJs() {
    // Execute the app.js code
    eval(appJsCode);

    // Return the functions we want to test
    return {
        switchTab,
        loadStats,
        searchPapers,
        displaySearchResults,
        showPaperDetails,
        sendChatMessage,
        addChatMessage,
        resetChat,
        showError,
        escapeHtml
    };
}

describe('NeurIPS Abstracts Web UI', () => {
    let app;

    beforeEach(() => {
        // Setup DOM
        document.body.innerHTML = `
            <div id="stats"></div>
            <button id="tab-search" class="tab-btn border-purple-600 text-gray-700"></button>
            <button id="tab-chat" class="tab-btn border-transparent text-gray-500"></button>
            <div id="search-tab" class="tab-content"></div>
            <div id="chat-tab" class="tab-content hidden"></div>
            <input id="search-input" value="" />
            <input id="use-embeddings" type="checkbox" />
            <select id="limit-select"><option value="10">10</option></select>
            <div id="search-results"></div>
            <input id="chat-input" value="" />
            <select id="n-papers"><option value="3">3</option></select>
            <div id="chat-messages"></div>
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
                min_year: 2020,
                max_year: 2025
            };

            fetch.mockResolvedValueOnce({
                json: async () => mockStats
            });

            await app.loadStats();

            const statsDiv = document.getElementById('stats');
            expect(statsDiv.innerHTML).toContain('12,500 Papers');
            expect(statsDiv.innerHTML).toContain('2020 - 2025');
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
                        id: 1,
                        name: 'Test Paper 1',
                        authors: ['Author A', 'Author B'],
                        abstract: 'This is a test abstract.',
                        session: 'Session 1A',
                        poster_position: '42',
                        paper_url: 'https://example.com/paper1'
                    },
                    {
                        id: 2,
                        name: 'Test Paper 2',
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
                papers: [{ id: 1, name: 'Test', authors: [], abstract: 'Test' }],
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
                        id: 1,
                        name: 'Test Paper',
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

        test('should use details element for long abstracts', () => {
            const longAbstract = 'A'.repeat(400);
            const data = {
                papers: [
                    {
                        id: 1,
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
            // Full abstract should be in the details content
            expect(resultsDiv.innerHTML).toContain('A'.repeat(400));
        });

        test('should not use details element for short abstracts', () => {
            const shortAbstract = 'This is a short abstract.';
            const data = {
                papers: [
                    {
                        id: 1,
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
                        id: 1,
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
            expect(resultsDiv.innerHTML).toContain('Relevance:');
            expect(resultsDiv.innerHTML).toContain('0.750');
        });

        test('should escape HTML in paper data', () => {
            const data = {
                papers: [
                    {
                        id: 1,
                        name: '<script>alert("xss")</script>',
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
            expect(resultsDiv.innerHTML).toContain('&lt;script&gt;');
            expect(resultsDiv.innerHTML).toContain('&lt;b&gt;Author&lt;/b&gt;');
            expect(resultsDiv.innerHTML).toContain('&lt;img');
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
            const useEmbeddings = document.getElementById('use-embeddings');
            const limitSelect = document.getElementById('limit-select');

            searchInput.value = 'neural networks';
            useEmbeddings.checked = true;

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

        test('should escape HTML in messages', () => {
            app.addChatMessage('<script>alert("xss")</script>', 'user');

            const messagesDiv = document.getElementById('chat-messages');
            expect(messagesDiv.innerHTML).not.toContain('<script>');
            expect(messagesDiv.innerHTML).toContain('&lt;script&gt;');
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
                id: 1,
                name: 'Test Paper',
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

            await app.showPaperDetails(1);

            expect(fetch).toHaveBeenCalledWith('/api/paper/1');

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
                id: 1,
                name: 'Test Paper',
                authors: [],
                abstract: 'Test'
            };

            fetch.mockResolvedValueOnce({
                json: async () => mockPaper
            });

            await app.showPaperDetails(1);

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
                id: 1,
                name: '<script>alert("xss")</script>',
                authors: ['<b>Author</b>'],
                abstract: '<img src=x onerror=alert(1)>'
            };

            fetch.mockResolvedValueOnce({
                json: async () => mockPaper
            });

            await app.showPaperDetails(1);

            const modal = document.querySelector('.fixed');
            expect(modal.innerHTML).not.toContain('<script>');
            expect(modal.innerHTML).toContain('&lt;script&gt;');
            expect(modal.innerHTML).toContain('&lt;b&gt;Author&lt;/b&gt;');
            expect(modal.innerHTML).toContain('&lt;img');
        });
    });
});
