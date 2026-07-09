import type { ProductMatch } from "../types";

function confidenceColor(confidence: string): string {
  const c = confidence.toLowerCase();
  if (c.includes("high")) return "#16a34a";
  if (c.includes("medium") || c.includes("mid")) return "#d97706";
  return "#dc2626";
}

export default function MatchCard({ title, match }: { title: string; match: ProductMatch }) {
  return (
    <details className="match-card" open={title === "Primary Match"}>
      <summary>
        {title} — {match.brand || "—"}
        <span style={{ color: confidenceColor(match.confidence) }}>
          {" "}
          ● {match.confidence || "—"}
        </span>
      </summary>
      <dl>
        <dt>Brand</dt>
        <dd>{match.brand || "—"}</dd>
        <dt>Products</dt>
        <dd>{match.products || "—"}</dd>
        <dt>Full Product Name</dt>
        <dd>{match.full_product_name || "—"}</dd>
        <dt>Product ID</dt>
        <dd>{match.product_id || "—"}</dd>
        <dt>SKU ID</dt>
        <dd>{match.sku_id || "—"}</dd>
        <dt>Product Variant ID</dt>
        <dd>{match.product_variant_id || "—"}</dd>
        <dt>Region</dt>
        <dd>{match.region || "—"}</dd>
        <dt>Species</dt>
        <dd>{match.species || "—"}</dd>
        <dt>Animal Size</dt>
        <dd>{match.animal_size || "—"}</dd>
        <dt>Pack Size</dt>
        <dd>{match.pack_size || "—"}</dd>
        <dt>Pack Quantity</dt>
        <dd>{match.pack_quantity || "—"}</dd>
        <dt>Reason</dt>
        <dd>{match.reason || "—"}</dd>
      </dl>
    </details>
  );
}
