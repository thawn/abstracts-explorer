/**
 * Markdown Rendering Utilities
 * 
 * This module provides utility functions for rendering markdown with LaTeX support.
 */

/**
 * Render inline markdown with LaTeX support (no block-level wrappers)
 * Use this for titles and other inline content where block elements like <p> are not desired.
 * @param {string} text - Inline markdown text to render
 * @returns {string} Rendered HTML without block-level wrappers
 */
export function renderInlineMarkdownWithLatex(text) {
    if (!text) return '';

    try {
        // Use parseInline for inline content to avoid wrapping in <p> tags
        return marked.parseInline(text);
    } catch (e) {
        console.warn('Markdown inline parsing error:', e);
        // Fallback to escaped HTML if markdown parsing fails
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

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
