/**
 * Security tests: markdown render helpers must sanitize XSS while preserving
 * legitimate markdown, KaTeX math, and #paper-N citation anchors.
 *
 * Unlike other suites, this one wires the REAL marked + marked-katex + DOMPurify
 * so it verifies actual sanitization behavior (not a passthrough mock).
 */

import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';
import DOMPurifyImport from 'dompurify';

// DOMPurify's default export is already an instance in a DOM env (jsdom);
// fall back to the factory form just in case.
const DOMPurify = typeof DOMPurifyImport.sanitize === 'function'
    ? DOMPurifyImport
    : DOMPurifyImport(window);

global.marked = marked;
global.markedKatex = markedKatex;
global.DOMPurify = DOMPurify;

import {
    renderMarkdownWithLatex,
    renderInlineMarkdownWithLatex,
    configureMarkedWithKatex,
} from '../static/modules/utils/markdown-utils.js';

describe('Markdown sanitization (XSS)', () => {
    it('strips event-handler attributes from images', () => {
        const html = renderMarkdownWithLatex('<img src=x onerror="alert(1)">');
        expect(html).not.toContain('onerror');
    });

    it('strips <script> tags', () => {
        const html = renderMarkdownWithLatex('hello <script>alert(1)</script> world');
        expect(html).not.toContain('<script>');
    });

    it('neutralizes javascript: URLs in links', () => {
        const html = renderMarkdownWithLatex('[click](javascript:alert(1))');
        expect(html).not.toContain('javascript:');
    });

    it('sanitizes inline-rendered titles too', () => {
        const html = renderInlineMarkdownWithLatex('<img src=x onerror="alert(1)">Title');
        expect(html).not.toContain('onerror');
    });

    it('preserves legitimate markdown formatting', () => {
        const html = renderMarkdownWithLatex('This is **bold** text');
        expect(html).toContain('<strong>');
        expect(html).toContain('bold');
    });

    it('preserves #paper-N citation anchors', () => {
        const html = renderMarkdownWithLatex('See [Paper-1](#paper-1)');
        expect(html).toContain('href="#paper-1"');
        expect(html).toContain('Paper-1');
    });

    it('preserves KaTeX math output', () => {
        configureMarkedWithKatex();
        const html = renderMarkdownWithLatex('Energy: $E=mc^2$');
        expect(html).toContain('katex');
    });
});
