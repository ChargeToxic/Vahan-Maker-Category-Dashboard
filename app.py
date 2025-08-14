import re
import io
import pandas as pd
import streamlit as st
import plotly.express as px

# ======================================================
# Page setup (simple, professional look)
# ======================================================
st.set_page_config(page_title="Vahan — Maker & Category Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Vahan Vehicle Registrations — Maker & Category")
st.caption("Investor-focused analytics: year range, category/manufacturer filters, trends & YoY growth.")

# ------------------------------------------------------
# Utility: detect header row & year from title line
# ------------------------------------------------------
def detect_header_row(raw_df: pd.DataFrame) -> int:
    """
    Find the row that contains the real table header (usually where LMV/MMV/HMV/TOTAL appear).
    """
    for i in range(min(15, len(raw_df))):
        row_vals = raw_df.iloc[i].astype(str).str.upper().tolist()
        text = " ".join([x for x in row_vals if x and x != "NAN"])
        if ("LMV" in text) and ("TOTAL" in text):
            return i
    return 4  # sensible fallback seen in your files

def detect_year_from_title(raw_df: pd.DataFrame):
    """
    Detect the year from the first cell text like '... (2025)'.
    """
    try:
        title = str(raw_df.iloc[0, 0])
        match = re.search(r"(20\d{2})", title)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None

# ------------------------------------------------------
# Cleaner for your "Maker Wise ... Vehicle Category Group" Excel
# ------------------------------------------------------
def clean_maker_file(file_obj) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with columns: ['Year','S No','Maker','LMV','MMV','HMV','TOTAL']
    Works with your uploaded 'reportTable' files.
    """
    # read raw first to detect header + year
    raw = pd.read_excel(file_obj, sheet_name=0, header=None, dtype=str)
    year = detect_year_from_title(raw)
    header_row = detect_header_row(raw)

    # read again with the detected header row
    df = pd.read_excel(file_obj, sheet_name=0, header=header_row)

    # Rename columns robustly (your files show these names after header=4)
    # Expected: ['Unnamed: 0','Unnamed: 1','4WIC','LMV','MMV','HMV','TOTAL']
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl.startswith("unnamed: 0"):
            rename_map[c] = "S No"
        elif cl.startswith("unnamed: 1"):
            rename_map[c] = "Maker"
        elif cl in ("4wic", "vehicle category group", "vehicle class"):
            rename_map[c] = "Vehicle Category Group"
        elif cl == "lmv":
            rename_map[c] = "LMV"
        elif cl == "mmv":
            rename_map[c] = "MMV"
        elif cl == "hmv":
            rename_map[c] = "HMV"
        elif cl in ("total", "tot"):
            rename_map[c] = "TOTAL"
    df = df.rename(columns=rename_map)

    # Keep only the columns we need
    keep = [c for c in ["S No", "Maker", "Vehicle Category Group", "LMV", "MMV", "HMV", "TOTAL"] if c in df.columns]
    df = df[keep].copy()

    # Drop empty rows & keep numeric S No if present
    df = df.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
    if "S No" in df.columns:
        df = df[pd.to_numeric(df["S No"], errors="coerce").notna()]

    # Numeric conversions for LMV/MMV/HMV/TOTAL (remove commas)
    for col in ["LMV", "MMV", "HMV", "TOTAL"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"\s+", "", regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Standardize Maker text
    if "Maker" in df.columns:
        df["Maker"] = df["Maker"].astype(str).str.strip().str.title()

    # Attach Year (from title)
    if year is None:
        # last resort: try to parse from filename (Streamlit uploads have .name)
        try:
            name = getattr(file_obj, "name", "")
            m = re.search(r"(20\d{2})", name)
            if m:
                year = int(m.group(1))
        except Exception:
            pass
    if year is None:
        raise ValueError("Could not detect year from the report — please ensure title contains (YYYY).")

    df["Year"] = year

    # Ensure all expected numeric columns exist
    for col in ["LMV", "MMV", "HMV", "TOTAL"]:
        if col not in df.columns:
            df[col] = 0

    # Reorder
    out_cols = ["Year", "S No", "Maker", "LMV", "MMV", "HMV", "TOTAL"]
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols].reset_index(drop=True)

# ------------------------------------------------------
# Sidebar: upload both files (your two reports)
# ------------------------------------------------------
st.sidebar.header("📂 Upload Vahan Maker Reports")
f1 = st.sidebar.file_uploader("Upload report (e.g., 2024)", type=["xlsx"])
f2 = st.sidebar.file_uploader("Upload report (e.g., 2025)", type=["xlsx"])

st.sidebar.caption("Tip: Use the exact 'Maker Wise Vehicle Category Group' exports.")

# ------------------------------------------------------
# Load & combine
# ------------------------------------------------------
if not f1 and not f2:
    st.info("Please upload at least one Excel report in the sidebar.")
    st.stop()

dfs = []
for f in [f1, f2]:
    if f:
        try:
            dfs.append(clean_maker_file(f))
        except Exception as e:
            st.error(f"Failed to read {getattr(f, 'name', 'file')}: {e}")

if not dfs:
    st.error("No valid report loaded.")
    st.stop()

data = pd.concat(dfs, ignore_index=True).sort_values(["Year", "Maker"]).reset_index(drop=True)

# ======================================================
# Filters
# ======================================================
st.sidebar.markdown("---")
years = sorted(data["Year"].unique())
year_min, year_max = min(years), max(years)
yr_from, yr_to = st.sidebar.select_slider("Year range", options=years, value=(year_min, year_max))

# vehicle class filter (LMV/MMV/HMV)
available_classes = [c for c in ["LMV", "MMV", "HMV"] if c in data.columns]
selected_classes = st.sidebar.multiselect("Vehicle classes", available_classes, default=available_classes)

# manufacturer filter
makers = sorted(data["Maker"].unique())
default_makers = makers if len(makers) <= 15 else makers[:15]
selected_makers = st.sidebar.multiselect("Manufacturers", makers, default=default_makers)

# primary view toggle
st.sidebar.markdown("---")
primary_view = st.sidebar.radio("Primary analysis by", options=["Manufacturer", "Vehicle Class"], index=0)

# Apply filters
mask = (data["Year"] >= yr_from) & (data["Year"] <= yr_to)
if selected_makers:
    mask &= data["Maker"].isin(selected_makers)

df = data.loc[mask].copy()

# If user restricted vehicle classes, recompute TOTAL from selected classes
if selected_classes and set(selected_classes) != set(available_classes):
    df["TOTAL"] = df[selected_classes].sum(axis=1)

if df.empty:
    st.warning("No data for the current filter selection.")
    st.stop()

# ======================================================
# KPIs
# ======================================================
agg_year = df.groupby("Year", as_index=False)["TOTAL"].sum().sort_values("Year")
latest_year = agg_year["Year"].max()
prev_year = agg_year["Year"].nlargest(2).min() if len(agg_year) > 1 else None

latest_total = int(agg_year.loc[agg_year["Year"] == latest_year, "TOTAL"].sum())
prev_total = int(agg_year.loc[agg_year["Year"] == prev_year, "TOTAL"].sum()) if prev_year else None
yoy_pct = ((latest_total - prev_total) / prev_total * 100) if prev_year and prev_total else None

top_maker_row = df.groupby(["Year", "Maker"], as_index=False)["TOTAL"].sum()
top_maker_row = top_maker_row.loc[top_maker_row["Year"] == latest_year].sort_values("TOTAL", ascending=False).head(1)
top_maker = top_maker_row["Maker"].iloc[0] if not top_maker_row.empty else "—"
top_maker_val = int(top_maker_row["TOTAL"].iloc[0]) if not top_maker_row.empty else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected total (latest year)", f"{latest_total:,}")
col2.metric("YoY change", f"{yoy_pct:.2f}%" if yoy_pct is not None else "N/A")
col3.metric("Top maker (latest year)", f"{top_maker}", f"{top_maker_val:,}")
col4.metric("Years in view", f"{yr_from} → {yr_to}")

st.markdown("---")

# ======================================================
# Trends & % change (YoY)
# ======================================================
st.subheader("📊 YoY % change by Manufacturer")

# Maker selector just for YoY comparison
yoy_maker_select = st.multiselect(
    "Select manufacturer(s) for YoY comparison",
    options=sorted(data["Maker"].unique()),
    default=sorted(data["Maker"].unique())[:5]
)

# Prepare YoY data by manufacturer
maker_year_all = data.groupby(["Year", "Maker"], as_index=False)["TOTAL"].sum()
maker_year_all = maker_year_all[maker_year_all["Maker"].isin(yoy_maker_select)]
maker_year_all = maker_year_all.sort_values(["Maker", "Year"])
maker_year_all["YoY %"] = maker_year_all.groupby("Maker")["TOTAL"].pct_change() * 100

fig_yoy_maker = px.bar(
    maker_year_all,
    x="Year", y="YoY %", color="Maker", barmode="group",
    text="YoY %",
    title="Year-over-Year % change (Manufacturer-wise)"
)
st.plotly_chart(fig_yoy_maker, use_container_width=True)

# ======================================================
# Maker vs Category comparison
# ======================================================
st.subheader("⚖️ Maker vs Category Analysis")

# Select one maker & one year
maker_for_cat = st.selectbox("Select manufacturer for category comparison", sorted(data["Maker"].unique()))
year_for_cat = st.selectbox("Select year for category comparison", sorted(data["Year"].unique()))

df_cat = data[(data["Maker"] == maker_for_cat) & (data["Year"] == year_for_cat)]

if not df_cat.empty:
    total_val = int(df_cat["TOTAL"].sum())
    category_vals = {
        "LMV": int(df_cat["LMV"].sum()) if "LMV" in df_cat else 0,
        "MMV": int(df_cat["MMV"].sum()) if "MMV" in df_cat else 0,
        "HMV": int(df_cat["HMV"].sum()) if "HMV" in df_cat else 0
    }

    # Pie chart for category composition
    fig_cat = px.pie(
        names=list(category_vals.keys()),
        values=list(category_vals.values()),
        title=f"{maker_for_cat} — Category breakdown ({year_for_cat})"
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown(f"**Total registrations:** {total_val:,}")
else:
    st.warning("No data available for this selection.")

# ======================================================
# Maker vs Category analysis
# ======================================================
if primary_view == "Manufacturer":
    st.subheader("🏭 Manufacturer performance")
    maker_year = df.groupby(["Year", "Maker"], as_index=False)["TOTAL"].sum()
    fig_maker = px.bar(maker_year, x="Maker", y="TOTAL", color="Year", barmode="group",
                       title="Manufacturer totals by year")
    st.plotly_chart(fig_maker, use_container_width=True)

    # YoY by maker (between adjacent years)
    yoy_maker = maker_year.sort_values(["Maker", "Year"]).copy()
    yoy_maker["YoY %"] = yoy_maker.groupby("Maker")["TOTAL"].pct_change() * 100
    latest_yoy = yoy_maker[yoy_maker["Year"] == latest_year].sort_values("YoY %", ascending=False)
    st.subheader(f"📈 YoY % by manufacturer — {latest_year}")
    st.dataframe(latest_yoy[["Maker", "Year", "TOTAL", "YoY %"]].reset_index(drop=True))

else:
    st.subheader("🚘 Vehicle class composition (LMV/MMV/HMV)")
    # Melt to long format for class breakdown
    long_cls = df.melt(id_vars=["Year", "Maker"], value_vars=available_classes, var_name="Class", value_name="Registrations")
    class_year = long_cls.groupby(["Year", "Class"], as_index=False)["Registrations"].sum()
    fig_class = px.bar(class_year, x="Year", y="Registrations", color="Class", barmode="stack",
                       title="Class composition by year")
    st.plotly_chart(fig_class, use_container_width=True)

    # Top maker per class (latest year)
    latest_cls = long_cls[long_cls["Year"] == latest_year]
    top_per_class = latest_cls.groupby(["Class", "Maker"], as_index=False)["Registrations"].sum()
    top_per_class = top_per_class.sort_values(["Class", "Registrations"], ascending=[True, False]).groupby("Class").head(5)
    st.subheader(f"🏆 Top makers by class — {latest_year}")
    st.dataframe(top_per_class.reset_index(drop=True))

# ======================================================
# Market share (latest year)
# ======================================================
st.subheader(f"🧭 Market share — {latest_year}")
latest = df[df["Year"] == latest_year]
share = latest.groupby("Maker", as_index=False)["TOTAL"].sum()
share["Share %"] = share["TOTAL"] / share["TOTAL"].sum() * 100
share = share.sort_values("TOTAL", ascending=False).head(15)
fig_share = px.pie(share, names="Maker", values="TOTAL", title=f"Market share (Top 15 makers) — {latest_year}")
st.plotly_chart(fig_share, use_container_width=True)

# ======================================================
# Data table & export
# ======================================================
st.subheader("🔎 Cleaned & filtered data")
st.dataframe(df.sort_values(["Year", "TOTAL"], ascending=[True, False]).reset_index(drop=True), use_container_width=True)

@st.cache_data
def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")

csv_all = to_csv_bytes(df)
st.download_button("⬇️ Download filtered data (CSV)", data=csv_all, file_name="vahan_filtered.csv", mime="text/csv")



