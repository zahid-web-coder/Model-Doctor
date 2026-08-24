"use client";

import type { RunMetrics } from "@/lib/compare/metrics";
import { winner } from "@/lib/compare/metrics";
import { isAvailable, type Value } from "@/lib/compare/provenance";

/**
 * The overall metric table.
 *
 * Three kinds of cell, visually distinct on purpose: a stored figure, a derived
 * one carrying the formula that produced it, and an unavailable one carrying
 * the reason it cannot be shown. An unavailable metric is never rendered as a
 * dash or a zero — both read as data.
 */

const pct = (v: number | null) => (v === null ? "—" : `${(v * 100).toFixed(1)}%`);
const num3 = (v: number | null) => (v === null ? "—" : v.toFixed(3));
const int = (v: number) => String(v);

function Cell({
  value,
  format,
  isWinner,
  note,
}: {
  value: Value<number | null> | Value<number>;
  format: (v: never) => string;
  isWinner: boolean;
  note: string | null;
}) {
  if (!isAvailable(value)) {
    return (
      <td className="py-2.5 px-3 text-right align-top">
        <span className="text-slate/70 text-[12px] italic">Not available</span>
        {note && <CellNote>{note}</CellNote>}
      </td>
    );
  }
  return (
    <td className="py-2.5 px-3 text-right align-top">
      <span
        className={`font-mono text-[13px] ${
          isWinner ? "text-brass font-semibold" : "text-ink"
        }`}
      >
        {format(value.value as never)}
      </span>
      {isWinner && <span className="text-brass text-[11px] ml-1.5">✓</span>}
      {note && <CellNote>{note}</CellNote>}
    </td>
  );
}

const CellNote = ({ children }: { children: string }) => (
  <span className="block text-[10px] text-slate/70 leading-snug mt-0.5">
    {children}
  </span>
);

/** The note a value carries: its reason if unavailable, its formula if derived. */
const noteOf = (value: Value<number | null> | Value<number>): string | null =>
  !isAvailable(value)
    ? value.reason
    : value.provenance === "derived"
      ? (value.formula ?? null)
      : null;

function Row({
  label,
  a,
  b,
  format,
  higherIsBetter,
}: {
  label: string;
  a: Value<number | null> | Value<number>;
  b: Value<number | null> | Value<number>;
  format: (v: never) => string;
  higherIsBetter: boolean;
}) {
  const win = winner(
    a as Value<number | null>,
    b as Value<number | null>,
    higherIsBetter,
  );

  // A note describes one run, not the row. Where both runs produce the same note
  // it is hoisted next to the metric name and stated once; where they differ it
  // must stay in its own cell. Mean IoU is the case that forces this — its
  // formula carries the number of stored measurements it averaged, and two runs
  // that found different numbers of objects average different sample sizes. One
  // shared note would put run A's denominator under run B's figure.
  const noteA = noteOf(a);
  const noteB = noteOf(b);
  const shared = noteA === noteB ? noteA : null;

  return (
    <tr className="border-b border-border/20 last:border-0">
      <td className="py-2.5 px-3 align-top">
        <span className="text-[13px] text-ink">{label}</span>
        {shared && (
          <span className="block text-[10px] text-slate/70 leading-snug mt-0.5 max-w-[380px]">
            {shared}
          </span>
        )}
      </td>
      <Cell value={a} format={format} isWinner={win === "a"} note={shared ? null : noteA} />
      <Cell value={b} format={format} isWinner={win === "b"} note={shared ? null : noteB} />
    </tr>
  );
}

export function MetricTable({
  labelA,
  labelB,
  a,
  b,
}: {
  labelA: string;
  labelB: string;
  a: RunMetrics;
  b: RunMetrics;
}) {
  return (
    <section className="bg-card border border-border/40 rounded-lg overflow-hidden">
      <header className="px-3 py-3 border-b border-border/40">
        <h3 className="text-[13px] font-medium text-ink">Overall</h3>
        <p className="text-[11px] text-slate mt-0.5">
          Stored counts, and figures derived from them. Nothing here comes from an
          offline evaluator.
        </p>
      </header>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-left text-slate border-b border-border/40 bg-canvas/40">
              <th className="font-medium text-[11px] py-2 px-3">Metric</th>
              <th className="font-medium text-[11px] py-2 px-3 text-right max-w-[180px] truncate">
                {labelA}
              </th>
              <th className="font-medium text-[11px] py-2 px-3 text-right max-w-[180px] truncate">
                {labelB}
              </th>
            </tr>
          </thead>
          <tbody>
            <Row label="Precision" a={a.precision} b={b.precision} format={pct} higherIsBetter />
            <Row label="Recall" a={a.recall} b={b.recall} format={pct} higherIsBetter />
            <Row label="F1" a={a.f1} b={b.f1} format={pct} higherIsBetter />
            <Row label="Mean box IoU" a={a.meanIou} b={b.meanIou} format={num3} higherIsBetter />
            <Row
              label="Mean mask IoU"
              a={a.meanMaskIou}
              b={b.meanMaskIou}
              format={num3}
              higherIsBetter
            />
            <Row label="Correct" a={a.correct} b={b.correct} format={int} higherIsBetter />
            <Row
              label="False negatives"
              a={a.falseNegative}
              b={b.falseNegative}
              format={int}
              higherIsBetter={false}
            />
            <Row
              label="False positives"
              a={a.falsePositive}
              b={b.falsePositive}
              format={int}
              higherIsBetter={false}
            />
            <Row
              label="Poor localisation"
              a={a.poorLocalization}
              b={b.poorLocalization}
              format={int}
              higherIsBetter={false}
            />
            <Row
              label="Wrong class"
              a={a.wrongClass}
              b={b.wrongClass}
              format={int}
              higherIsBetter={false}
            />

            {/* Last, and deliberately so. These four rows are unavailable for
                every run, and sitting mid-table they separated the derived rates
                from the stored counts with a block of things the page cannot
                answer. Ordering the table derived → stored → unavailable keeps
                them visible and honest without letting them lead. */}
            <Row label="mAP@50 (box)" a={a.mapBox50} b={b.mapBox50} format={num3} higherIsBetter />
            <Row
              label="mAP@50:95 (box)"
              a={a.mapBox5095}
              b={b.mapBox5095}
              format={num3}
              higherIsBetter
            />
            <Row label="mAP@50 (mask)" a={a.mapMask50} b={b.mapMask50} format={num3} higherIsBetter />
            <Row
              label="mAP@50:95 (mask)"
              a={a.mapMask5095}
              b={b.mapMask5095}
              format={num3}
              higherIsBetter
            />
          </tbody>
        </table>
      </div>
    </section>
  );
}
