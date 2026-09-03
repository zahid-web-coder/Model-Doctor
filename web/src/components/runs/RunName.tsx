"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Pencil, Loader2 } from "lucide-react";
import { control, ControlError } from "@/lib/api/control";

/**
 * A run's name, editable in place.
 *
 * **The name is the only writable field on a run.** Everything else records
 * what an analysis pass did, and rewriting any of it would leave the stored
 * parameters disagreeing with the findings derived from them. What a run is
 * *called* belongs to the reader: Ultralytics writes every checkpoint to
 * `best.pt`, so runs 4, 5, 7 and 10 are all "best__3_.pt" on the same dataset
 * and split, and only their ids tell them apart.
 *
 * **An unnamed run is blank.** Filling the gap with the model file or the id
 * would make "named" and "unnamed" indistinguishable at a glance, which is the
 * confusion this exists to end, and a column of "Add a name" prompts is
 * instructions where data should be. The pencil on hover says the cell is
 * writable without a word in every row.
 *
 * The write goes to the control API. The reader opens SQLite read-only and
 * declares no non-GET route, and a rename does not get to be the exception.
 */
export function RunName({
  runId,
  name,
}: {
  runId: number;
  name: string | null;
}) {
  const router = useRouter();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(name ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /**
   * Begin editing from the name as it currently stands.
   *
   * Seeded here rather than kept in sync by an effect: a name changed
   * elsewhere — another tab, or the header — would otherwise be overwritten by
   * whatever this component last held.
   */
  function begin() {
    setDraft(name ?? "");
    setError(null);
    setEditing(true);
  }

  async function save() {
    const next = draft.trim();
    if (next === (name ?? "")) {
      setEditing(false);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await control.renameRun(runId, next || null);
      setEditing(false);
      // The runs list is a server component; refresh re-reads it rather than
      // mutating a local copy that could then disagree with the database.
      router.refresh();
    } catch (cause) {
      setError(
        cause instanceof ControlError
          ? cause.message
          : "The analysis service is not reachable, so the name was not saved."
      );
    } finally {
      setBusy(false);
    }
  }

  if (!editing) {
    return (
      <button
        type="button"
        onClick={begin}
        title={name ? "Rename this run" : "Name this run"}
        className="group inline-flex items-center gap-2 text-left max-w-full"
      >
        {/* An unnamed run is blank, not a prompt: a column of "Add a name"
            reads as instructions where data should be. */}
        <span className="truncate text-[12px] text-ink group-hover:text-brass transition-colors">
          {name}
        </span>
        {/* **The pencil is always drawn, not revealed on hover.** A control
            that appears only once the pointer is over it cannot be discovered
            by looking, and on a blank cell there is nothing else to suggest
            the column can be written at all. Quiet enough not to compete with
            the names beside it, and it is the whole cell that is clickable. */}
        <span className="shrink-0 w-[22px] h-[22px] grid place-items-center rounded border border-border/50 text-slate/70 group-hover:border-brass/50 group-hover:text-brass transition-colors">
          <Pencil size={11} />
        </span>
      </button>
    );
  }

  return (
    <span className="inline-flex flex-col gap-1">
      <span className="inline-flex items-center gap-1.5">
        <input
          value={draft}
          autoFocus
          // Selected on focus rather than in an effect, so the whole existing
          // name is replaced by typing without a second render pass.
          onFocus={(e) => e.currentTarget.select()}
          disabled={busy}
          maxLength={80}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") save();
            // Escape abandons the edit rather than saving it, and stops there:
            // a lightbox or dialog above this must not also close because a
            // reader changed their mind about a name.
            if (e.key === "Escape") {
              e.stopPropagation();
              setDraft(name ?? "");
              setEditing(false);
            }
          }}
          onBlur={save}
          placeholder="Name this run"
          aria-label={`Name for run ${runId}`}
          className="w-[190px] px-2 py-1 rounded border border-brass/50 bg-canvas text-[12px] text-ink outline-none focus:border-brass"
        />
        {busy && <Loader2 size={12} className="animate-spin text-slate" />}
      </span>
      {error && <span className="text-[10px] text-[#A65C48]">{error}</span>}
    </span>
  );
}
