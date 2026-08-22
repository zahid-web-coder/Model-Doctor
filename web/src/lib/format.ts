/**
 * Rendering helpers for values the API may not have.
 *
 * A null from the backend means "not measured", and showing 0 instead turns an
 * absent measurement into a real one — which is the single most damaging thing
 * a diagnostics UI can do. Everything nullable goes through here.
 */

export const NOT_MEASURED = "n/a";

export function num(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return NOT_MEASURED;
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return NOT_MEASURED;
  return `${(value * 100).toFixed(digits)}%`;
}

export function lift(value: number | null | undefined): string {
  if (value === null || value === undefined) return NOT_MEASURED;
  return `${value.toFixed(2)}x`;
}

export function pValue(value: number | null | undefined): string {
  if (value === null || value === undefined) return NOT_MEASURED;
  return value < 0.001 ? "<0.001" : value.toFixed(3);
}

/** A predicted class the schema cannot supply is shown as absent, not guessed. */
export function predictedClass(value: string | null | undefined): string {
  return value ?? NOT_MEASURED;
}
