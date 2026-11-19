/**
 * Lazy Plotly loader.
 *
 * Plotly is the largest frontend dependency, so it is loaded as a separate
 * chunk only when a chart is first needed (Clusters tab or a chat
 * visualization). Tests inject a global ``Plotly`` mock, which is preferred so
 * the dynamic import is skipped.
 */

let loaded: any = null;
let loadPromise: Promise<any> | null = null;

/**
 * Return the loaded Plotly instance, preferring a test-injected global.
 *
 * @throws Error when Plotly has not been loaded yet.
 */
export function getPlotly(): any {
    const impl = (globalThis as any).Plotly || loaded;
    if (!impl) {
        throw new Error('Plotly has not been loaded yet. Call ensurePlotly() before rendering.');
    }
    return impl;
}

/**
 * Ensure Plotly is loaded, importing the chunk on first use.
 *
 * @returns The Plotly instance.
 */
export function ensurePlotly(): Promise<any> {
    const globalImpl = (globalThis as any).Plotly;
    if (globalImpl) {
        return Promise.resolve(globalImpl);
    }
    if (loaded) {
        return Promise.resolve(loaded);
    }
    if (!loadPromise) {
        loadPromise = import('plotly.js-basic-dist-min').then((module) => {
            loaded = (module as any).default ?? module;
            return loaded;
        });
    }
    return loadPromise;
}
