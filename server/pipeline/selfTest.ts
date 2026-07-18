import { readFileSync } from "node:fs";
import { join } from "node:path";
import { rootDir } from "../config.js";
import { analyseFasta } from "./analyse.js";

function assert(condition: unknown, message: string): void {
  if (!condition) throw new Error(message);
}

const failSample = readFileSync(join(rootDir, "demo_samples", "likely_fail_sample.fasta"), "utf8");
const failResult = analyseFasta(failSample, "likely_fail_sample.fasta", "Escherichia coli");
const ampicillin = failResult.predictions.find((prediction) => prediction.antibiotic === "Ampicillin");

assert(ampicillin?.decision === "likely_to_fail", "Ampicillin should be likely_to_fail for blaTEM-1 sample.");
assert(failResult.markers.some((marker) => marker.symbol === "blaTEM-1"), "Expected blaTEM-1 marker.");

const unsupported = analyseFasta(failSample, "likely_fail_sample.fasta", "Unsupported species");
assert(unsupported.predictions.every((prediction) => prediction.decision === "no_call"), "Unsupported species must force no-call.");

console.log("Pipeline self-tests passed.");
