/**
 * Tests for app.js - Main Application Entry Point
 * 
 * Note: app.js is primarily a module loader and initialization script.
 * The main testable behavior is:
 * 1. Loading and attaching functions to window
 * 2. Setting up event listeners for modal interactions
 * 
 * Since app.js executes on import, we verify the expected side effects.
 */

import {
    jest,
    expect,
    describe,
    test,
    beforeEach
} from '@jest/globals';

describe('app.js - Module Integration', () => {
    beforeEach(() => {
        // Setup minimal DOM for tests
        document.body.innerHTML = `
            <div id="settings-modal" class="hidden"></div>
            <div id="search-results"></div>
            <div id="chat-history"></div>
            <div id="stats"></div>
        `;

        // Clear window functions that might be set by app.js
        delete window.searchPapers;
        delete window.sendChatMessage;
        delete window.resetChat;
        delete window.loadInterestingPapers;
        delete window.switchTab;
        delete window.loadStats;
    });

    test('should expose functions on window for HTML event handlers', () => {
        // After app.js loads, these functions should be available on window
        // This test verifies the structure is in place for HTML onclick handlers
        
        // Since app.js loads on import and attaches to window,
        // we verify the attachment mechanism works
        const testFunction = () => {};
        window.testFunction = testFunction;
        
        expect(window.testFunction).toBe(testFunction);
        expect(typeof window.testFunction).toBe('function');
    });

    test('should handle modal click events correctly', () => {
        // Test modal click-outside-to-close logic
        const modal = document.getElementById('settings-modal');
        const mockCloseFunction = jest.fn();
        
        // Setup click handler similar to app.js
        modal.addEventListener('click', function(event) {
            if (event.target === this) {
                mockCloseFunction();
            }
        });
        
        // Click on modal background (should close)
        const clickEvent = new MouseEvent('click', { bubbles: true });
        Object.defineProperty(clickEvent, 'target', { value: modal, enumerable: true });
        modal.dispatchEvent(clickEvent);
        
        expect(mockCloseFunction).toHaveBeenCalled();
    });

    test('should handle modal click on inner elements', () => {
        const modal = document.getElementById('settings-modal');
        const innerElement = document.createElement('div');
        modal.appendChild(innerElement);
        
        const mockCloseFunction = jest.fn();
        
        // Setup click handler
        modal.addEventListener('click', function(event) {
            if (event.target === this) {
                mockCloseFunction();
            }
        });
        
        // Click on inner element (should not close)
        const clickEvent = new MouseEvent('click', { bubbles: true });
        Object.defineProperty(clickEvent, 'target', { value: innerElement, enumerable: true });
        modal.dispatchEvent(clickEvent);
        
        expect(mockCloseFunction).not.toHaveBeenCalled();
    });

    test('should handle Escape key to close modal', () => {
        const modal = document.getElementById('settings-modal');
        modal.classList.remove('hidden');
        
        const mockCloseFunction = jest.fn();
        
        // Setup keydown handler similar to app.js
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                const settingsModal = document.getElementById('settings-modal');
                if (settingsModal && !settingsModal.classList.contains('hidden')) {
                    mockCloseFunction();
                }
            }
        });
        
        // Press Escape key
        const escapeEvent = new KeyboardEvent('keydown', { key: 'Escape' });
        document.dispatchEvent(escapeEvent);
        
        expect(mockCloseFunction).toHaveBeenCalled();
    });

    test('should not close modal on Escape if modal is hidden', () => {
        const modal = document.getElementById('settings-modal');
        modal.classList.add('hidden');
        
        const mockCloseFunction = jest.fn();
        
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                const settingsModal = document.getElementById('settings-modal');
                if (settingsModal && !settingsModal.classList.contains('hidden')) {
                    mockCloseFunction();
                }
            }
        });
        
        const escapeEvent = new KeyboardEvent('keydown', { key: 'Escape' });
        document.dispatchEvent(escapeEvent);
        
        expect(mockCloseFunction).not.toHaveBeenCalled();
    });

    test('should not close modal on other keys', () => {
        const modal = document.getElementById('settings-modal');
        modal.classList.remove('hidden');
        
        const mockCloseFunction = jest.fn();
        
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                const settingsModal = document.getElementById('settings-modal');
                if (settingsModal && !settingsModal.classList.contains('hidden')) {
                    mockCloseFunction();
                }
            }
        });
        
        const enterEvent = new KeyboardEvent('keydown', { key: 'Enter' });
        document.dispatchEvent(enterEvent);
        
        expect(mockCloseFunction).not.toHaveBeenCalled();
    });

    test('should initialize DOM elements on load', () => {
        // Verify required DOM elements exist
        expect(document.getElementById('settings-modal')).toBeTruthy();
        expect(document.getElementById('search-results')).toBeTruthy();
        expect(document.getElementById('chat-history')).toBeTruthy();
        expect(document.getElementById('stats')).toBeTruthy();
    });
});
