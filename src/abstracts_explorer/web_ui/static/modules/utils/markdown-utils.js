/**
 * Markdown Rendering Utilities
 * 
 * This module provides utility functions for rendering markdown with LaTeX support.
 */

/**
 * Render markdown with LaTeX support
 * @param {string} text - Markdown text to render
 * @returns {string} Rendered HTML
 */
export function renderMarkdownWithLatex(text) {
    if (!text) return '';
    
    try {
        // Use the globally loaded marked library with KaTeX extension
        return marked.parse(text);
    } catch (e) {
        console.warn('Markdown parsing error:', e);
        // Fallback to escaped HTML if markdown parsing fails
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }
}

/**
 * Configure marked with KaTeX extension
 * Should be called once during initialization
 */
export function configureMarkedWithKatex() {
    if (typeof markedKatex !== 'undefined' && typeof marked !== 'undefined') {
        marked.use(markedKatex({
            throwOnError: false,
            nonStandard: true,
            output: 'html'
        }));
    }
}
