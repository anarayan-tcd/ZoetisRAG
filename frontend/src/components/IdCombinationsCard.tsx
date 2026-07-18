import type { IdCombination } from "../types";

interface IdCombinationsCardProps {
  combinations: IdCombination[];
  reason: string;
}

function packLabel(combo: IdCombination): string {
  if (combo.pack_units) return `${combo.pack_units} units`;
  if (combo.ml_volume) return `${combo.ml_volume}ml`;
  return "—";
}

// The ids the matched product corresponds to. Plural on purpose: the same
// real product is routinely listed under several duplicate/legacy catalog
// rows that nothing in the data distinguishes, so all of them are shown
// rather than one being picked arbitrarily.
export default function IdCombinationsCard({ combinations, reason }: IdCombinationsCardProps) {
  const isAmbiguous = combinations.length > 1;

  return (
    <section className="id-card">
      <header className="id-card-head">
        <h3>Possible ID Combinations</h3>
        <span className={`id-count-pill${isAmbiguous ? " ambiguous" : ""}`}>
          {combinations.length === 0
            ? "none found"
            : combinations.length === 1
              ? "1 match"
              : `${combinations.length} possible`}
        </span>
      </header>

      {combinations.length === 0 ? (
        <p className="id-empty">No catalog IDs correspond to this product.</p>
      ) : (
        <div className="id-table-wrap">
          <table className="id-table">
            <thead>
              <tr>
                <th>Product ID</th>
                <th>Variant ID</th>
                <th>SKU ID</th>
                <th>Pack</th>
                <th>Catalog Description</th>
              </tr>
            </thead>
            <tbody>
              {combinations.map((c, i) => (
                <tr key={`${c.product_id}-${c.product_variant_id}-${c.sku_id}-${i}`}>
                  <td className="id-num">{c.product_id || "—"}</td>
                  <td className="id-num">{c.product_variant_id || "—"}</td>
                  <td className="id-num">{c.sku_id || "—"}</td>
                  <td className="id-num">{packLabel(c)}</td>
                  <td className="id-desc">
                    {c.catalog_description || "—"}
                    {c.occurrences > 0 && (
                      <span className="id-occ"> · prescribed {c.occurrences.toLocaleString()}×</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {reason && <p className="id-reason">{reason}</p>}
    </section>
  );
}
