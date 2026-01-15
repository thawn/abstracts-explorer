/**
 * Jest setup file for DOM testing
 */

import { jest } from '@jest/globals';
import '@testing-library/jest-dom';

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
