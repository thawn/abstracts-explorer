/**
 * Markdown Rendering Utilities
 *
 * Renders markdown (with LaTeX) and sanitizes the resulting HTML with DOMPurify
 * before it is inserted into the DOM, preventing XSS from untrusted content
 * (scraped paper abstracts/titles and LLM chat output).
 */

import { marked as markedLib } from 'marked';
import DOMPurifyLib from 'dompurify';
import markedKatexLib from 'marked-katex-extension';

// Tests inject global mocks; prefer them when present, otherwise use the
// bundled library. Access is lazy so globals set after import are honoured.
function getMarked(): typeof markedLib {
    return ((globalThis as any).marked as typeof markedLib) || markedLib;
}

function getDOMPurify(): typeof DOMPurifyLib {
    return ((globalThis as any).DOMPurify as typeof DOMPurifyLib) || DOMPurifyLib;
}

function getMarkedKatex(): typeof markedKatexLib {
    return ((globalThis as any).markedKatex as typeof markedKatexLib) || markedKatexLib;
}

/**
 * Sanitize an HTML string with DOMPurify.
 *
 * Uses DOMPurify's secure defaults, which strip <script>, on* event-handler
 * attributes, and javascript:/data: script URLs while preserving standard
 * markup, <a href="#paper-N"> citation anchors, and KaTeX (output: 'html')
 * spans (class/style/aria-hidden are retained).
 */
function sanitizeHtml(html: string): string {
    return getDOMPurify().sanitize(html);
}

/**
 * Render inline markdown with LaTeX support (no block-level wrappers).
 */
export function renderInlineMarkdownWithLatex(text: string | null | undefined): string {
    if (!text) return '';

    try {
        return sanitizeHtml(getMarked().parseInline(text) as string);
    } catch (e) {
        console.warn('Markdown inline parsing error:', e);
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

/**
 * Render markdown with LaTeX support.
 */
export function renderMarkdownWithLatex(text: string | null | undefined): string {
    if (!text) return '';

    try {
        return sanitizeHtml(getMarked().parse(text) as string);
    } catch (e) {
        console.warn('Markdown parsing error:', e);
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }
}

/**
 * Configure marked with the KaTeX extension.
 * Should be called once during initialization.
 */
export function configureMarkedWithKatex(): void {
    getMarked().use(
        getMarkedKatex()({
            throwOnError: false,
            nonStandard: true,
            output: 'html'
        })
    );
}
