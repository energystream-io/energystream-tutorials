# -*- coding: utf-8 -*-
"""
Anson 1: hail-storm production and cost-of-outage analysis, using only the
public EnergyStream API (https://energystream.io/api/v1) - no internal
database, no Athena, no pre-exported CSVs. Anyone with an EnergyStream API
key can run this end to end and get the same charts and numbers.

Requires a Professional or Enterprise API key. Every endpoint this script
calls except /facilities/{plant_id}/iso-plant and
/facilities/{plant_id}/iso-node-relationships needs the 'ercot' or 'models'
capability, which is Professional and up and NOT part of the free trial for
these specific endpoints - a Free-plan key will get a 403 on most of this.
See https://energystream.io/docs/api and the interactive reference at
https://energystream.io/api-reference.

Set your key as an environment variable before running:
    export ENERGYSTREAM_API_KEY=es_live_...      (macOS/Linux)
    $env:ENERGYSTREAM_API_KEY = "es_live_..."     (Windows PowerShell)

    pip install requests pandas matplotlib
    python anson-analysis.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import pandas as pd
import requests

BASE_URL = os.environ.get("ENERGYSTREAM_API_BASE_URL", "https://energystream.io/api/v1")
PLANT_ID = "11724"  # Anson Solar Center, Jones County, TX

API_KEY = os.environ.get("ENERGYSTREAM_API_KEY")
if not API_KEY:
    sys.exit(
        "Set ENERGYSTREAM_API_KEY first. Generate a key from your "
        "EnergyStream profile page (Professional or Enterprise plan)."
    )

SESSION = requests.Session()
SESSION.headers.update({"Authorization": f"Bearer {API_KEY}"})


def call(path, params=None):
    """GET one /api/v1 endpoint and return its `data` object. Raises with a
    plain-English explanation on anything other than 200 - a 403 here almost
    always means the key's plan doesn't include this endpoint, not a bug in
    this script."""
    resp = SESSION.get(f"{BASE_URL}{path}", params=params)
    if resp.status_code == 403:
        raise RuntimeError(
            f"{path} -> 403. This endpoint needs a Professional or "
            f"Enterprise plan; a Free-plan key cannot reach it. "
            f"Response: {resp.text}"
        )
    if resp.status_code == 429:
        raise RuntimeError(f"{path} -> 429 rate limited: {resp.text}")
    resp.raise_for_status()
    return resp.json()["data"]


def fetch_all_rows(path, params, start, end):
    """GET every row of a windowed/time-series endpoint across `start` to
    `end`, inclusive, handling the per-request-plan cap on history length
    (the response's `window.max_days`) by walking forward in that many
    days per call. A null `max_days` (Enterprise) means the whole range
    came back in one call."""
    rows = []
    cursor = start
    while cursor <= end:
        data = call(path, {**params, "start": iso(cursor), "end": iso(end)})
        rows.extend(data["rows"])
        max_days = (data.get("window") or {}).get("max_days")
        if not max_days:
            break
        cursor = cursor + timedelta(days=max_days)
    return rows


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def require_rows(rows, what):
    """Exit with a plain message instead of a cryptic pandas KeyError a few
    lines later. Trying this script against a different plant_id is exactly
    what it suggests doing - a plant with no data for one of these datasets
    yet is a normal, expected response (200, empty rows), not a bug."""
    if not rows:
        sys.exit(
            f"No {what} for plant_id={PLANT_ID} over this date range. "
            f"Try a different plant_id - see /facilities for the full list."
        )


UTC = timezone.utc
# Two versions of each date, used in two genuinely different contexts:
#   - the _utc ones are real UTC instants, used for the API's start/end
#     params and for comparisons against other UTC-aware timestamps.
#   - the plain ones are timezone-naive, used against the daily-rollup
#     indexes below, which are deliberately naive too: a day there means a
#     LOCAL (America/Chicago) calendar date, not a UTC instant, so slicing
#     and plotting against them has to use a naive date as well, or pandas
#     refuses the comparison outright (as it should - "May 2nd Central time"
#     and "May 2nd UTC" are different 24-hour windows).
hail_damage_date_utc = pd.Timestamp("2024-05-02", tz="UTC")
outage_end_fallback_utc = pd.Timestamp("2024-12-31", tz="UTC")  # last hail-related outage
hail_damage_date = pd.Timestamp("2024-05-02")
outage_end_fallback = pd.Timestamp("2024-12-31")
NOW = datetime.now(UTC)


# ---------------------------------------------------------------------------
# Step 1: Plant Metadata - which ISO unit(s) and node(s) does this plant have?
# GET /facilities/{plant_id}/iso-plant and /facilities/{plant_id}/iso-node-relationships
# ---------------------------------------------------------------------------
print("Fetching ISO unit(s) for plant", PLANT_ID, "...")
iso_plant_rows = call(f"/facilities/{PLANT_ID}/iso-plant")["rows"]
iso_plants = pd.DataFrame(iso_plant_rows)
print(iso_plants[["iso_id", "unit_code", "iso", "installed_capacity_mw"]])

# node_id values for financials/settlement-prices are available via
# /facilities/{plant_id}/iso-node-relationships.
print("\nFetching ISO node relationship(s) ...")
node_rows = call(f"/facilities/{PLANT_ID}/iso-node-relationships")["rows"]
node_relations = pd.DataFrame(node_rows)
print(node_relations[["iso_id", "resource_node"]])

unit_codes = iso_plants["unit_code"].dropna().unique().tolist()
node_ids = node_relations["iso_id"].dropna().unique().tolist()

# Commissioning date, so the full-history pulls below don't start earlier
# than the plant has ever had data.
generators = pd.DataFrame(call(f"/facilities/{PLANT_ID}/generators")["rows"])
commissioned = pd.to_datetime(
    generators["operating_year_month"].dropna().min() + "-01"
).tz_localize("UTC")
history_start = min(commissioned, hail_damage_date_utc)


# ---------------------------------------------------------------------------
# Step 2: Full production history - actual (ERCOT-reported) vs. ideal (modeled)
# GET /facilities/{plant_id}/production/interval
# GET /facilities/{plant_id}/models/iso-generation-estimates
# ---------------------------------------------------------------------------
print(f"\nFetching actual production from {history_start.date()} to {NOW.date()} ...")
actual_rows = fetch_all_rows(
    f"/facilities/{PLANT_ID}/production/interval", {}, history_start, NOW
)
require_rows(actual_rows, "actual production data")
actual = pd.DataFrame(actual_rows)
actual["timestamp_utc"] = pd.to_datetime(actual["timestamp_utc"], utc=True)

print("Fetching modeled (PVWatts ISO-calibrated) generation for the same range ...")
ideal_rows = fetch_all_rows(
    f"/facilities/{PLANT_ID}/models/iso-generation-estimates", {}, history_start, NOW
)
require_rows(ideal_rows, "modeled generation data")
ideal = pd.DataFrame(ideal_rows)
ideal["timestamp_utc"] = pd.to_datetime(ideal["timestamp_utc"], utc=True)

# Roll up to local (Central) operating days, one column per unit, to
# replicate the old wide-format daily rollup CSV without needing one.
def to_daily_by_unit(df, value_col="ac_energy_mwh"):
    local_date = (
        df["timestamp_utc"].dt.tz_convert("America/Chicago").dt.tz_localize(None).dt.normalize()
    )
    return df.assign(date=local_date).groupby(["date", "unit_code"])[value_col].sum().unstack("unit_code")


actual_daily_by_unit = to_daily_by_unit(actual.merge(iso_plants[["iso_id", "unit_code"]], on="iso_id"))
ideal_daily_by_unit = to_daily_by_unit(ideal)

# Drop the pre-commissioning zeros so the axis starts at first generation.
first_gen = actual_daily_by_unit[actual_daily_by_unit.sum(axis=1) > 0].index.min()
actual_daily_by_unit = actual_daily_by_unit.loc[first_gen:]

outage_end = outage_end_fallback
unit_labels = {code: f"Unit {i + 1}" for i, code in enumerate(sorted(actual_daily_by_unit.columns))}

fig, axes = plt.subplots(len(unit_labels), 1, figsize=(11, 6.5), sharex=True, sharey=True)
if len(unit_labels) == 1:
    axes = [axes]
for ax, (code, name) in zip(axes, unit_labels.items()):
    s = actual_daily_by_unit[code]
    ax.plot(s.index, s, color="#2a6fb0", alpha=0.25, lw=0.8, label="Daily")
    ax.plot(s.index, s.rolling(30, center=True, min_periods=15).mean(),
            color="#2a6fb0", lw=2, label="30-day average")
    ax.axvspan(hail_damage_date, outage_end, color="#d9622b", alpha=0.10, lw=0)
    ax.axvline(hail_damage_date, color="#222222", lw=1, ls="--")
    ax.set_ylabel("MWh per day", color="#666666")
    ax.set_title(name, loc="left", fontsize=11)
    ax.grid(axis="y", color="#e6e6e6")
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].annotate("Hail storm 2 May 2024\n(shaded: reported outages)",
                  xy=(hail_damage_date, 1), xycoords=("data", "axes fraction"),
                  xytext=(8, -6), textcoords="offset points", va="top", fontsize=9)
axes[0].legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
fig.suptitle("Anson 1: Daily Generation by Unit (ERCOT reported)", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig("anson_daily_rollup.png", dpi=200)
plt.show()


# ---------------------------------------------------------------------------
# Step 3: Actual vs. ideal, summed across both units
# ---------------------------------------------------------------------------
actual_daily = actual_daily_by_unit.sum(axis=1).rename("actual_mwh")
ideal_daily = ideal_daily_by_unit.sum(axis=1).rename("ideal_mwh")
cmp = pd.concat([actual_daily, ideal_daily], axis=1).dropna()
pct_ideal = (
    cmp["actual_mwh"].rolling(30, center=True, min_periods=15).sum()
    / cmp["ideal_mwh"].rolling(30, center=True, min_periods=15).sum() * 100
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for col, color, label in [
    ("ideal_mwh", "#555555", "Ideal (PVWatts estimate)"),
    ("actual_mwh", "#2a6fb0", "Actual (ERCOT reported)"),
]:
    ax1.plot(cmp.index, cmp[col], color=color, alpha=0.2, lw=0.8)
    ax1.plot(cmp.index, cmp[col].rolling(30, center=True, min_periods=15).mean(),
              color=color, lw=2, label=label + ", 30-day avg")
ax2.plot(pct_ideal.index, pct_ideal, color="#2a6fb0", lw=2)
ax2.axhline(100, color="#555555", lw=1, ls=":")
ax1.set_ylabel("MWh per day (both units)", color="#666666")
ax2.set_ylabel("% of ideal (30-day)", color="#666666")
ax1.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
for ax in (ax1, ax2):
    ax.axvspan(hail_damage_date, outage_end, color="#d9622b", alpha=0.10, lw=0)
    ax.axvline(hail_damage_date, color="#222222", lw=1, ls="--")
    ax.grid(axis="y", color="#e6e6e6")
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
ax1.annotate("Hail storm 2 May 2024\n(shaded: reported outages)",
             xy=(hail_damage_date, 1), xycoords=("data", "axes fraction"),
             xytext=(8, -6), textcoords="offset points", va="top", fontsize=9)
fig.suptitle("Anson 1: Actual vs. Ideal Daily Production, Both Units Combined", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig("anson_actual_vs_ideal.png", dpi=200)
plt.show()

outage = cmp.loc[hail_damage_date:outage_end]
print(f"\nHail outage window: actual {outage['actual_mwh'].sum():,.0f} MWh vs. "
      f"ideal {outage['ideal_mwh'].sum():,.0f} MWh "
      f"({outage['actual_mwh'].sum() / outage['ideal_mwh'].sum():.0%} of ideal)")

in_outage = (cmp.index >= hail_damage_date) & (cmp.index <= outage_end)
for label, subset in [
    ("Excluding outage period", cmp[~in_outage]),
    ("Before storm", cmp[cmp.index < hail_damage_date]),
    ("After outage period", cmp[cmp.index > outage_end]),
]:
    print(f"{label}: actual {subset['actual_mwh'].sum():,.0f} MWh vs. "
          f"ideal {subset['ideal_mwh'].sum():,.0f} MWh "
          f"({subset['actual_mwh'].sum() / subset['ideal_mwh'].sum():.1%} of ideal)")

win_start, win_end = (hail_damage_date.month, hail_damage_date.day), (12, 31)
rows = {}
for year in sorted(cmp.index.year.unique()):
    window = cmp.loc[f"{year}-{win_start[0]:02d}-{win_start[1]:02d}":f"{year}-12-31"]
    expected_days = (
        pd.Timestamp(f"{year}-12-31") - pd.Timestamp(f"{year}-{win_start[0]:02d}-{win_start[1]:02d}")
    ).days + 1
    if len(window) < expected_days:
        continue  # partial year
    rows[year] = {"actual_mwh": window["actual_mwh"].sum(), "ideal_mwh": window["ideal_mwh"].sum()}
by_year = pd.DataFrame(rows).T
by_year["pct_of_ideal"] = by_year["actual_mwh"] / by_year["ideal_mwh"] * 100
print("\nMay 2 - Dec 31 production by year:")
print(by_year.round(1))

hail_year = hail_damage_date.year
other = by_year.drop(index=hail_year)
hail = by_year.loc[hail_year]
print(f"\n{hail_year} vs. other years (same May 2 - Dec 31 window, {', '.join(map(str, other.index))}):")
print(f"  Actual MWh: {hail['actual_mwh'] / other['actual_mwh'].mean() - 1:+.1%} vs. the average of other years")
print(f"  % of ideal: {hail['pct_of_ideal']:.1f}% vs. {other['pct_of_ideal'].mean():.1f}% "
      f"({hail['pct_of_ideal'] - other['pct_of_ideal'].mean():+.1f} pts)")


# ---------------------------------------------------------------------------
# Step 4: Cost of the outage - lost energy x settlement price, per unit
# GET /facilities/{plant_id}/outages
# GET /facilities/{plant_id}/financials/settlement-prices
# ---------------------------------------------------------------------------
# NOTE: 'COLD WEATHER RELATED' is copied as-is from the original query this
# script replaces. It does not obviously match a May hailstorm - verify this
# is the right `reason` value for the outage you're investigating before
# trusting the cost figure below. See the outages endpoint's docs for the
# full set of outage_type/reason values this dataset actually uses.
print("\nFetching outage events ...")
outage_rows = call(
    f"/facilities/{PLANT_ID}/outages",
    {"start": iso(pd.to_datetime("2024-01-01").tz_localize("UTC"))},
)["rows"]
outages_df = pd.DataFrame(outage_rows)
outages_df["start_timestamp_utc"] = pd.to_datetime(outages_df["start_timestamp_utc"], utc=True)
outages_df["end_timestamp_utc"] = pd.to_datetime(outages_df["end_timestamp_utc"], utc=True)
outages_df = outages_df[outages_df["reason"] == "COLD WEATHER RELATED"]

if outages_df.empty:
    print("No outages matched the 'COLD WEATHER RELATED' filter - skipping cost analysis.")
else:
    outages_df["curtailment_threshold_mw"] = (
        outages_df["resource_pmax_mw"].astype(float) - outages_df["curtailment_mw"].astype(float)
    )
    start_outages = outages_df["start_timestamp_utc"].min()
    end_outages = outages_df["end_timestamp_utc"].max()

    ideal_window = ideal[
        (ideal["timestamp_utc"] >= start_outages) & (ideal["timestamp_utc"] <= end_outages)
    ].copy()
    ideal_window["ac_power_mw"] = ideal_window["ac_energy_mwh"] * 4

    actual_window = actual.merge(iso_plants[["iso_id", "unit_code"]], on="iso_id")
    actual_window = actual_window.rename(columns={"ac_energy_mwh": "actual_energy_mwh"})

    combo_prod = pd.merge(
        ideal_window, actual_window[["timestamp_utc", "unit_code", "actual_energy_mwh"]],
        on=["timestamp_utc", "unit_code"], how="left",
    )
    combo_prod["lost_energy_mwh"] = (
        combo_prod["ac_energy_mwh"] - combo_prod["actual_energy_mwh"]
    ).clip(lower=0)

    print("SUMMED lost energy:")
    print(combo_prod["lost_energy_mwh"].sum())

    print(f"Fetching settlement point prices for node(s) {node_ids} ...")
    spp_frames = []
    for node_id in node_ids:
        rows = fetch_all_rows(
            f"/facilities/{PLANT_ID}/financials/settlement-prices",
            {"node_id": node_id}, start_outages.to_pydatetime(), end_outages.to_pydatetime(),
        )
        if rows:
            spp_frames.append(pd.DataFrame(rows))
        else:
            # Not every ISO node has settlement-price data yet - this is a
            # normal, expected response (200 with an empty rows array), not
            # an error. Anson's own node has none as of this writing.
            print(f"  node_id {node_id}: no settlement-price data available, skipping.")

    if not spp_frames:
        print(
            "No settlement-price data available for any of this plant's node(s) - "
            "cannot compute a dollar cost for the outage. Showing lost energy "
            "(MWh) only; try a different plant_id to see the cost calculation."
        )
    else:
        spp = pd.concat(spp_frames, ignore_index=True)
        spp["interval_start"] = pd.to_datetime(spp["interval_start"], utc=True)

        cost_generation_estimates = pd.merge(
            combo_prod, spp, left_on="timestamp_utc", right_on="interval_start", how="left"
        )
        cost_generation_estimates["spp"] = cost_generation_estimates["spp"].ffill()
        cost_generation_estimates["lost_energy_cost"] = (
            cost_generation_estimates["lost_energy_mwh"] * cost_generation_estimates["spp"]
        )

        print("SUMMED ENERGY COST:")
        print(cost_generation_estimates["lost_energy_cost"].sum())
