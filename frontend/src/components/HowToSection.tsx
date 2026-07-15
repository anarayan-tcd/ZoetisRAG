import { useEffect, useState } from "react";

export default function HowToSection() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;

    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }

    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [open]);

  return (
    <>
      <button type="button" className="how-to-trigger" onClick={() => setOpen(true)}>
        How To Use This Tool
      </button>

      {open && (
        <div className="how-to-overlay" onClick={() => setOpen(false)}>
          <div className="how-to-modal" onClick={(e) => e.stopPropagation()}>
            <div className="how-to-modal-header">
              <h2>How To Use This Tool</h2>
              <button
                type="button"
                className="how-to-close"
                onClick={() => setOpen(false)}
                aria-label="Close"
                title="Close"
              >
                ✕
              </button>
            </div>

            <div className="how-to-content">
              <h3>Search Types</h3>
              <dl>
                <dt>Single Query</dt>
                <dd>
                  Paste one prescription line (e.g. "Frontline Spray 100ml dog") and get its
                  best-matching product back immediately.
                </dd>
                <dt>Batch Upload</dt>
                <dd>
                  Upload a CSV or TXT file with one prescription line per row (CSV needs a header row).
                  Every row is matched and you can download the full results as CSV, Excel, or JSON.
                </dd>
              </dl>

              <h3>Output Headers and Variables</h3>
              <dl>
                <dt>Primary Match / Alternatives</dt>
                <dd>
                  The Primary Match is the system's best guess. The two Alternatives are other
                  plausible products, ranked by likelihood if there is any ambiguity in the input.
                </dd>
                <dt>Brand / Products / Full Product Name</dt>
                <dd>
                  Brand is the manufacturer brand (e.g. "Frontline"). Products is just the short
                  product line (e.g. "Spot On", "Spray"). Full Product
                  Name is the complete variant name including species, size, and pack count (e.g.
                  "Spot On Medium Dog (10-20kg) 6 Pipette Pack").
                </dd>
                <dt>Product ID / SKU ID / Product Variant ID</dt>
                <dd>
                  Identifiers from historical prescription records, filled in ONLY when a
                  historical example confidently matches. Left blank rather than guessed when unsure.
                  This comes from examples given in Labelled.csv, and as this is not a very reliable source,
                  SKUID and Variant ID can be left blank even when the product is known
                </dd>
                <dt>Region</dt>
                <dd>
                  Which market the matched product variant is sold in (e.g. UK/IE, US), as the same
                  brand is sometimes sold under different names in different regions.
                </dd>
              </dl>

              <h3>Confidence scores (VERY IMPORTANT)</h3>
              <dl>
                <dt>High</dt>
                <dd>The product (or an unambiguous abbreviation of it) was named explicitly.</dd>
                <dt>Medium</dt>
                <dd>Strongly implied by ingredient, packaging language, or a partial name match.</dd>
                <dt>Low</dt>
                <dd>Best-guess from vague details only, with no direct product signal.</dd>
                <dt>Very Low</dt>
                <dd>None of the known candidates plausibly match the input.</dd>
              </dl>

              <h3>Optional checkboxes</h3>
              <dl>
                <dt>Enable live web research fallback</dt>
                <dd>
                  When nothing matches confidently, researches the product live via a Gemini web
                  search. Costs real API tokens/money and adds latency — off by default.
                </dd>
                <dt>Enable LLM-as-judge review</dt>
                <dd>
                  Runs a second, independent DeepSeek check on High/Medium-confidence matches against
                  real historical records, downgrading confidence if something doesn't line up (e.g. a
                  species mismatch). An extra LLM call per qualifying result — roughly +50% more calls
                  on average — so it costs more and takes longer. Off by default.
                </dd>
              </dl>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
