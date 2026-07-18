import type { ProductMatch } from "../types";

function confidenceColor(confidence: string): string {
  const c = confidence.toLowerCase();
  if (c.includes("high")) return "#16a34a";
  if (c.includes("medium") || c.includes("mid")) return "#d97706";
  return "#dc2626";
}

// The product only — no ids. Which catalog rows this product corresponds
// to is a separate question with a separate answer (often several), and
// IdCombinationsCard owns it.
export default function MatchCard({ match }: { match: ProductMatch }) {
  return (
    <section className="match-card">
      <header className="match-card-head">
        <h3>Matched Product</h3>
        <span className="confidence-pill" style={{ color: confidenceColor(match.confidence) }}>
          ● {match.confidence || "—"}
        </span>
      </header>
      <dl>
        <dt>Brand</dt>
        <dd>{match.brand || "—"}</dd>
        <dt>Products</dt>
        <dd>{match.products || "—"}</dd>
        <dt>Full Product Name</dt>
        <dd>{match.full_product_name || "—"}</dd>
        <dt>Species</dt>
        <dd>{match.species || "—"}</dd>
        <dt>Animal Size</dt>
        <dd>{match.animal_size || "—"}</dd>
        <dt>Pack Size</dt>
        <dd>{match.pack_size || "—"}</dd>
        <dt>Pack Quantity</dt>
        <dd>{match.pack_quantity || "—"}</dd>
        <dt>Region</dt>
        <dd>{match.region || "—"}</dd>
        <dt>Reason</dt>
        <dd>{match.reason || "—"}</dd>
      </dl>
    </section>
  );
}
