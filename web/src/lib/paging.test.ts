import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { pageWindow, resolvePageSize, PAGE_SIZES } from "./paging.ts";

/**
 * The pager used to render "first five, then the last" regardless of where the
 * reader actually was, so from page six onward the current page had no button
 * and nothing was highlighted — the control contradicted the table beside it.
 * The property that matters is therefore not the exact layout but that the
 * current page is always present, which is what most of these assert.
 */
describe("pageWindow", () => {
  test("always contains the current page", () => {
    for (const count of [1, 2, 7, 27, 300]) {
      for (let page = 1; page <= count; page += 1) {
        assert.ok(
          pageWindow(page, count).includes(page),
          `page ${page} of ${count} was not offered`
        );
      }
    }
  });

  test("always offers the first and last page", () => {
    const window = pageWindow(14, 27);
    assert.ok(window.includes(1), "first page unreachable");
    assert.ok(window.includes(27), "last page unreachable");
  });

  test("the regression it was written for: page 14 of 27", () => {
    // The old rule produced [1,2,3,4,5,"gap",27] here — no 14 anywhere.
    assert.deepEqual(pageWindow(14, 27), [1, "gap", 13, 14, 15, "gap", 27]);
  });

  test("no gap stands in for a single page", () => {
    // A "…" that hides exactly one number is wider than the number it hides.
    for (const count of [5, 6, 7, 8, 27]) {
      for (let page = 1; page <= count; page += 1) {
        const window = pageWindow(page, count);
        window.forEach((entry, i) => {
          if (entry !== "gap") return;
          const before = window[i - 1];
          const after = window[i + 1];
          if (typeof before === "number" && typeof after === "number") {
            assert.ok(
              after - before > 2,
              `gap hid only page ${before + 1} (${page}/${count})`
            );
          }
        });
      }
    }
  });

  test("stays ordered and free of duplicates", () => {
    const numbers = pageWindow(5, 27).filter((p): p is number => p !== "gap");
    assert.deepEqual(numbers, [...numbers].sort((a, b) => a - b));
    assert.equal(new Set(numbers).size, numbers.length);
  });

  test("degenerate counts do not produce buttons that cannot be pressed", () => {
    assert.deepEqual(pageWindow(1, 1), [1]);
    assert.deepEqual(pageWindow(1, 0), []);
  });

  test("a short list is shown in full, with no gaps at all", () => {
    assert.deepEqual(pageWindow(2, 4), [1, 2, 3, 4]);
  });
});

/**
 * The page size becomes a `limit` on an API request, so anything a query string
 * can hold has to survive this without being forwarded.
 */
describe("resolvePageSize", () => {
  test("accepts the sizes actually offered", () => {
    for (const size of PAGE_SIZES) {
      assert.equal(resolvePageSize(String(size)), size);
    }
  });

  test("falls back to the default for anything else", () => {
    for (const raw of [undefined, "", "0", "-5", "11", "1000000", "abc", "10.5", "1e9"]) {
      assert.equal(
        resolvePageSize(raw),
        PAGE_SIZES[0],
        `"${raw}" was not rejected`
      );
    }
  });
});
