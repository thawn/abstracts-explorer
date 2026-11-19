// Lightweight Plotly stand-in for Jest/JSdom. Delegates to a global Plotly
// mock when tests provide one, otherwise returns no-op functions.
const Plotly = new Proxy(
    {},
    {
        get(_target, prop) {
            const impl = globalThis.Plotly;
            if (impl && prop in impl) {
                const value = impl[prop];
                return typeof value === 'function' ? value.bind(impl) : value;
            }
            return () => undefined;
        },
    }
);

export default Plotly;
