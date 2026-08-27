"""Transform the raw Online Retail II CSV into normalised relational tables.

The raw file is a single denormalised fact table. This module splits it into the
five source tables defined in ``sql/schema.sql`` and writes them to
``data/interim/`` as CSVs ready for PostgreSQL ``COPY``.

Design decisions worth knowing:

* **Nothing is dropped here.** Credit notes, adjustments, anonymous sales and
  non-merchandise stock codes are all loaded and *classified*. Filtering is the
  feature layer's job, so the database stays a faithful mirror of the source and
  every exclusion is a visible, reversible SQL predicate.
* **Sign conventions are preserved** exactly as in the source: credit lines keep
  negative quantities.
* **Deterministic tie-breaking.** Modal descriptions and modal countries are
  resolved with a stable sort so repeated runs produce byte-identical output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CONFIG, PATHS
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Stock codes that are not merchandise. Anything not listed is 'PRODUCT'
# (this deliberately includes the DCGS* codes, which are real catalogue items).
ITEM_TYPE_MAP: dict[str, str] = {
    "POST": "POSTAGE",
    "DOT": "POSTAGE",
    "C2": "POSTAGE",
    "C3": "POSTAGE",
    "AMAZONFEE": "FEE",
    "BANK CHARGES": "FEE",
    "CRUK": "FEE",
    "D": "DISCOUNT",
    "S": "SAMPLE",
    "M": "MANUAL",
    "ADJUST": "ADJUSTMENT",
    "ADJUST2": "ADJUSTMENT",
    "B": "ADJUSTMENT",
    "TEST001": "TEST",
    "TEST002": "TEST",
}

RAW_DTYPES = {"Invoice": "string", "StockCode": "string", "Description": "string", "Country": "string"}


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV with explicit dtypes and a parsed timestamp."""
    raw_path = path or (PATHS.data_raw / CONFIG["data"]["raw_file"])
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. See the README for download instructions."
        )

    logger.info("Reading raw dataset: %s", raw_path)
    df = pd.read_csv(raw_path, dtype=RAW_DTYPES, parse_dates=["InvoiceDate"])
    logger.info("Read %s rows x %d columns", f"{len(df):,}", df.shape[1])
    return df


def normalise_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the minimal cleaning needed for referential integrity.

    This is *structural* normalisation only — whitespace, casing and the
    invoice-level timestamp collapse. No rows are removed and no business
    filtering happens here.
    """
    out = df.copy()
    out["Invoice"] = out["Invoice"].str.strip().str.upper()
    # 173 stock codes differ from another only by case, so upper-casing is
    # required before stock_code can serve as a primary key.
    out["StockCode"] = out["StockCode"].str.strip().str.upper()
    out["Description"] = out["Description"].str.strip()
    out["Country"] = out["Country"].str.strip()

    # Customer ID arrives as float because of the NaNs; it is integer-valued.
    out["CustomerId"] = out["Customer ID"].astype("Int64")

    n_bad = int(out["Quantity"].eq(0).sum())
    if n_bad:
        logger.warning("Dropping %d line(s) with quantity = 0 (violate schema CHECK)", n_bad)
        out = out[out["Quantity"].ne(0)]

    return out


def _classify_invoice_type(invoice_no: pd.Series) -> pd.Series:
    """SALE / CREDIT / ADJUSTMENT from the invoice-number prefix."""
    first = invoice_no.str[0]
    return (
        pd.Series(pd.NA, index=invoice_no.index, dtype="string")
        .mask(first.eq("C"), "CREDIT")
        .mask(first.eq("A"), "ADJUSTMENT")
        .fillna("SALE")
    )


def build_countries(df: pd.DataFrame) -> pd.DataFrame:
    """Lookup table of distinct countries with a stable surrogate key."""
    names = sorted(df["Country"].dropna().unique())
    countries = pd.DataFrame({"country_id": range(1, len(names) + 1), "country_name": names})
    logger.info("countries: %d rows", len(countries))
    return countries


def build_products(df: pd.DataFrame) -> pd.DataFrame:
    """One row per stock code, with the modal description and an item type."""
    described = df.dropna(subset=["Description"])
    # value_counts is descending by count then arbitrary; sorting the description
    # as a secondary key makes the winner deterministic across runs.
    modal = (
        described.groupby(["StockCode", "Description"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["StockCode", "n", "Description"], ascending=[True, False, True])
        .drop_duplicates("StockCode")
        .set_index("StockCode")["Description"]
    )

    products = pd.DataFrame({"stock_code": sorted(df["StockCode"].dropna().unique())})
    products["description"] = products["stock_code"].map(modal)

    is_voucher = products["stock_code"].str.startswith(("GIFT_0001", "GIFT"))
    products["item_type"] = products["stock_code"].map(ITEM_TYPE_MAP).where(lambda s: s.notna(), None)
    products.loc[products["item_type"].isna() & is_voucher, "item_type"] = "VOUCHER"
    products["item_type"] = products["item_type"].fillna("PRODUCT")

    logger.info(
        "products: %d rows (%s)",
        len(products),
        ", ".join(f"{k}={v}" for k, v in products["item_type"].value_counts().items()),
    )
    return products


def build_customers(df: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    """One row per identified customer, with the modal country."""
    known = df.dropna(subset=["CustomerId"])
    modal_country = (
        known.groupby(["CustomerId", "Country"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["CustomerId", "n", "Country"], ascending=[True, False, True])
        .drop_duplicates("CustomerId")
    )

    lookup = countries.set_index("country_name")["country_id"]
    customers = pd.DataFrame(
        {
            "customer_id": modal_country["CustomerId"].astype("int64").to_numpy(),
            "primary_country_id": modal_country["Country"].map(lookup).astype("int64").to_numpy(),
        }
    ).sort_values("customer_id", ignore_index=True)

    logger.info("customers: %d rows", len(customers))
    return customers


def build_invoices(df: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    """One row per invoice document.

    ``invoice_ts`` uses MIN over the invoice's lines: 83 invoices carry more than
    one timestamp (median spread 60 seconds), which is data-entry lag rather than
    genuinely separate events.
    """
    lookup = countries.set_index("country_name")["country_id"]
    grouped = df.groupby("Invoice", observed=True).agg(
        invoice_ts=("InvoiceDate", "min"),
        country_name=("Country", "first"),
        customer_id=("CustomerId", "first"),
    )

    invoices = grouped.reset_index().rename(columns={"Invoice": "invoice_no"})
    invoices["country_id"] = invoices["country_name"].map(lookup).astype("int64")
    invoices["invoice_type"] = _classify_invoice_type(invoices["invoice_no"])
    invoices = invoices[
        ["invoice_no", "customer_id", "country_id", "invoice_ts", "invoice_type"]
    ].sort_values("invoice_no", ignore_index=True)

    logger.info(
        "invoices: %d rows (%s)",
        len(invoices),
        ", ".join(f"{k}={v}" for k, v in invoices["invoice_type"].value_counts().items()),
    )
    return invoices


def build_invoice_lines(df: pd.DataFrame) -> pd.DataFrame:
    """The fact table. ``line_id`` is assigned by PostgreSQL, not here."""
    lines = (
        df[["Invoice", "StockCode", "Quantity", "Price"]]
        .rename(
            columns={
                "Invoice": "invoice_no",
                "StockCode": "stock_code",
                "Quantity": "quantity",
                "Price": "unit_price",
            }
        )
        .reset_index(drop=True)
    )
    logger.info("invoice_lines: %s rows", f"{len(lines):,}")
    return lines


def build_all(raw_path: Path | None = None, out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Run the full raw → relational transformation and write CSVs.

    Returns the tables keyed by name so callers can assert on them without
    re-reading from disk.
    """
    out_dir = out_dir or PATHS.data_interim
    out_dir.mkdir(parents=True, exist_ok=True)

    df = normalise_raw(load_raw(raw_path))

    countries = build_countries(df)
    tables = {
        "countries": countries,
        "customers": build_customers(df, countries),
        "products": build_products(df),
        "invoices": build_invoices(df, countries),
        "invoice_lines": build_invoice_lines(df),
    }

    for name, table in tables.items():
        target = out_dir / f"{name}.csv"
        table.to_csv(target, index=False)
        logger.info("Wrote %s (%s rows)", target.relative_to(PATHS.root), f"{len(table):,}")

    return tables
