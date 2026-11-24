import * as esbuild from "esbuild";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("..", import.meta.url));
const watch = process.argv.includes("--watch");

const staticDir = path.join(root, "src/abstracts_explorer/web_ui/static");

const options = {
    entryPoints: [path.join(staticDir, "app.ts")],
    outdir: path.join(staticDir, "dist"),
    bundle: true,
    format: "esm",
    splitting: true,
    entryNames: "app",
    chunkNames: "chunks/[name]-[hash]",
    sourcemap: watch,
    minify: !watch,
    target: ["es2022"],
    logLevel: "info",
    absWorkingDir: root,
    define: {
        "process.env.NODE_ENV": JSON.stringify(watch ? "development" : "production"),
    },
};

if (watch) {
    const context = await esbuild.context(options);
    await context.watch();
    console.log("Watching web UI TypeScript sources for changes...");
} else {
    await esbuild.build(options);
}
