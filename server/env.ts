import { join } from "node:path";

// Load .env (gitignored) into process.env if present. Node >= 20.12 provides
// process.loadEnvFile; real environment variables always take precedence.
try {
  (process as unknown as { loadEnvFile: (p: string) => void }).loadEnvFile(join(process.cwd(), ".env"));
} catch {
  /* no .env file — rely on the ambient environment */
}
