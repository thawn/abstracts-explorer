/**
 * Jest setup file for DOM testing
 */

// Add custom matchers from @testing-library/jest-dom
require('@testing-library/jest-dom');

// Mock fetch globally
global.fetch = jest.fn();

// Mock console methods to reduce noise in tests
global.console = {
    ...console,
    error: jest.fn(),
    warn: jest.fn(),
};

// Setup DOM helpers
beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();

    // Reset fetch mock
    fetch.mockReset();

    // Clear document body
    document.body.innerHTML = '';
});
