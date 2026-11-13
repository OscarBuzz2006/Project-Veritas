import { cp, mkdir } from "node:fs/promises";

// Build output only: preserve the Python report without editing docs/.
const source = new URL("../../docs/", import.meta.url);
const destination = new URL("../dist/legacy/", import.meta.url);
await mkdir(destination, { recursive: true });
await cp(source, destination, { recursive: true });
console.log("Standalone Python report staged at dist/legacy/");
