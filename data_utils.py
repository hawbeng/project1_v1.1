"""Loading, cleaning and summary helpers for the Singapore Jobs Explorer.

The same functions feed the Streamlit app (app.py) and the EDA notebook
(notebooks/01_eda.ipynb), so a number in the notebook and the same number in
the dashboard always come from one piece of code.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants: every judgement call lives here, in one place, with its reason
# ---------------------------------------------------------------------------

DATE_CANDIDATES = (
    "metadata_newPostingDate",
    "metadata_originalPostingDate",
    "posted_date",
    "date_posted",
)

# Posting VOLUME is only complete from this month (collection ramped up before it).
TREND_CUTOFF = pd.Timestamp("2023-05-01")

# Views and applications stop accumulating for adverts first posted after June 2023
# (mean views per posting fall from ~111 to ~5). Competition and interest metrics
# therefore use only adverts first posted up to this date.
ENGAGEMENT_END = pd.Timestamp("2023-06-30")

# A monthly salary outside this range is not a monthly wage (annual figures typed
# into a monthly field, or placeholders). The salary is blanked; the posting is kept.
SALARY_FLOOR = 500
SALARY_CEILING = 60_000

# minimumYearsExperience above this is treated as a typing error.
MAX_EXPERIENCE_YEARS = 40

SENIORITY_BANDS = {
    "Fresh/entry level": "Entry",
    "Non-executive": "Entry",
    "Junior Executive": "Junior",
    "Executive": "Mid",
    "Professional": "Mid",
    "Senior Executive": "Senior",
    "Manager": "Management",
    "Middle Management": "Management",
    "Senior Management": "Management",
}
SENIORITY_ORDER = ["Entry", "Junior", "Mid", "Senior", "Management"]

# PMET = Professionals, Managers, Executives and Technicians. The source has no PMET
# field, so we use a simple rule: only "Non-executive" postings are Non-PMET.
NON_PMET_LEVELS = {"Non-executive"}

# The 43 posting categories grouped into 7 broad sectors, so a user can start from
# "which part of the economy" before choosing a specific category.
BROAD_SECTORS = {
    "Finance & Professional Services": [
        "Accounting / Auditing / Taxation", "Banking and Finance", "Insurance", "Legal",
        "Consulting", "Professional Services", "Risk Management",
        "Real Estate / Property Management",
    ],
    "Technology & Media": [
        "Information Technology", "Telecommunications", "Advertising / Media", "Design",
        "Marketing / Public Relations", "Entertainment",
    ],
    "Industry & Infrastructure": [
        "Engineering", "Building and Construction", "Manufacturing", "Precision Engineering",
        "Repair and Maintenance", "Logistics / Supply Chain", "Architecture / Interior Design",
        "Environment / Health", "Sciences / Laboratory / R&D",
    ],
    "Public Service, Education & Health": [
        "Healthcare / Pharmaceutical", "Medical / Therapy Services", "Education and Training",
        "Public / Civil Service", "Social Services", "Security and Investigation",
    ],
    "Consumer & Lifestyle": [
        "F&B", "Hospitality", "Sales / Retail", "Personal Care / Beauty", "Travel / Tourism",
        "Events / Promotions", "Wholesale Trade", "Purchasing / Merchandising", "Customer Service",
    ],
    "Admin, HR & Management": [
        "Admin / Secretarial", "Human Resources", "General Management", "General Work",
    ],
    "Others": ["Others"],
}
CATEGORY_TO_SECTOR = {cat: sector for sector, cats in BROAD_SECTORS.items() for cat in cats}
SECTOR_ORDER = list(BROAD_SECTORS)

# One colour per broad sector, shared by every chart in the app.
SECTOR_COLOURS = {
    "Finance & Professional Services": "#1F5F8B",
    "Technology & Media": "#2F6FB0",
    "Industry & Infrastructure": "#2F8F83",
    "Public Service, Education & Health": "#C2185B",
    "Consumer & Lifestyle": "#E07A2E",
    "Admin, HR & Management": "#8E7CC3",
    "Others": "#8B95A1",
}

SKILL_KEYWORDS = {
    "Python": r"\bpython\b",
    "SQL": r"\bsql\b",
    "Java": r"\bjava\b",
    ".NET": r"\.net\b",
    "AWS": r"\baws\b",
    "Azure": r"\bazure\b",
    "SAP": r"\bsap\b",
    "Excel": r"\bexcel\b",
    "Cloud": r"\bcloud\b",
    "Data": r"\bdata\b",
    "Cybersecurity": r"cyber|security",
    "Machine learning / AI": r"machine learning|\bai\b|\bml\b",
    "Salesforce": r"salesforce",
    "React": r"\breact\b",
    "Project management": r"project manag",
    "Accounting": r"account",
    "Digital marketing": r"digital market",
    "Nursing": r"\bnurs",
}


# ---------------------------------------------------------------------------
# Reading and small helpers
# ---------------------------------------------------------------------------

def read_jobs(source, nrows: int | None = None) -> pd.DataFrame:
    """Read a path or Streamlit UploadedFile as CSV."""
    return pd.read_csv(source, low_memory=False, nrows=nrows)


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    return next((name for name in candidates if name in columns), None)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _category_names(value) -> list[str]:
    """Every category a posting belongs to, in source order."""
    if isinstance(value, list):
        items = value
    elif value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    else:
        try:
            items = json.loads(str(value))
        except (json.JSONDecodeError, TypeError):
            text = str(value).strip()
            return [text] if text else []
    names = []
    for item in items or []:
        name = (item.get("category") or item.get("name")) if isinstance(item, dict) else str(item)
        if name:
            names.append(str(name).strip())
    return names


# Public name used by the notebook.
parse_categories = _category_names


def _category_name(value) -> str | None:
    names = _category_names(value)
    return names[0] if names else None


def _parse_dates(df: pd.DataFrame, column: str) -> pd.Series:
    """Parse dates, reading ISO dates (2023-04-08) as year-month-day.

    The course file uses ISO dates. Parsing them with dayfirst=True would read
    2024-01-05 as 1 May 2024, so ISO strings are parsed explicitly first and only
    the remaining formats (e.g. 08/04/2023) are read day-first.
    """
    if column not in df:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    text = df[column].astype("string").str.strip()
    iso = text.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    parsed = pd.to_datetime(text.where(iso).str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
    other = text.notna() & ~iso
    if other.any():
        parsed = parsed.fillna(pd.to_datetime(text.where(other), errors="coerce", dayfirst=True))
    return parsed


def clean_title(titles: pd.Series) -> pd.Series:
    """Strip recruitment marketing from job titles so one role is counted once.

    "Urgent Hiring!!! Business Development Manager (MES) - Up to $9K"
    becomes "Business Development Manager".
    """
    t = titles.astype("string")
    # Leading symbols, then hiring slogans such as "Urgent Hiring!!!" or "2x".
    t = t.str.replace(r"^[^A-Za-z0-9]+", "", regex=True)
    slogan = r"(?i)^(?:(?:urgent(?:ly)?|immediate(?:ly)?|now|we\s+are|hiring|wanted|needed|\d+\s*x)\b[\s!:,\-]*)+"
    t = t.str.replace(slogan, "", regex=True)
    # Everything after a " | " separator (locations, perks, levels).
    t = t.str.replace(r"\s*\|.*$", "", regex=True)
    # Bracketed detail: "(Java, CIM)", "[Contract]".
    t = t.str.replace(r"[\(\[\{][^\)\]\}]*[\)\]\}]?", " ", regex=True)
    # Pay phrases at the end: "- Up to $9K", "$3,000 - $4,000".
    t = t.str.replace(r"(?i)[\s,\-–]*(?:up\s*to\b|\$|s\$|sgd\b).*$", "", regex=True)
    t = t.str.replace(r"\s+", " ", regex=True).str.strip(" -–,:/.")
    return t.mask(t == "")


# ---------------------------------------------------------------------------
# Cleaning and features
# ---------------------------------------------------------------------------

def clean_jobs(raw: pd.DataFrame, return_audit: bool = False):
    """Return a cleaned copy with derived features, keeping unrecognised columns.

    With return_audit=True, also return a dict counting what each rule changed,
    so the cleaning can be reported rather than taken on trust.
    """
    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]
    audit = {"rows_in": len(df)}

    if "title" not in df and "job_title" in df:
        df["title"] = df["job_title"]

    # 1. Rows with no job id are entirely empty (3,988 in the full file): drop them.
    if "metadata_jobPostId" in df:
        empty = df["metadata_jobPostId"].isna()
        df = df.loc[~empty].copy()
        audit["dropped_empty_rows"] = int(empty.sum())

    # 2. Categories: parse the JSON once per distinct value, not once per row.
    if "categories" in df:
        distinct = df["categories"].dropna().unique()
        lookup = {value: _category_names(value) for value in distinct}
        df["category_list"] = df["categories"].map(lambda v: lookup.get(v, []) if isinstance(v, str) else [])
    elif "category" in df:
        df["category_list"] = df["category"].map(lambda v: [v] if isinstance(v, str) and v else [])
    else:
        df["category_list"] = [[] for _ in range(len(df))]
    df["category"] = df["category_list"].map(lambda names: names[0] if names else None).astype("string")

    # 3. Dates. posted_date = latest posting date; original_date = first posting date.
    new_date = _parse_dates(df, "metadata_newPostingDate")
    df["original_date"] = _parse_dates(df, "metadata_originalPostingDate")
    fallback = _first_existing(df.columns, ("posted_date", "date_posted"))
    extra = _parse_dates(df, fallback) if fallback else pd.Series(pd.NaT, index=df.index)
    df["posted_date"] = new_date.fillna(df["original_date"]).fillna(extra)
    df["original_date"] = df["original_date"].fillna(df["posted_date"])
    audit["unparseable_dates"] = int(df["posted_date"].isna().sum())

    # 4. Salary: zero means "not disclosed"; swap reversed ranges; blank impossible values.
    for column in ("salary_minimum", "salary_maximum", "average_salary"):
        if column in df:
            df[column] = _numeric(df, column)
    has_bounds = {"salary_minimum", "salary_maximum"}.issubset(df.columns)
    if has_bounds:
        zero = (df["salary_minimum"] <= 0) | (df["salary_maximum"] <= 0)
        audit["salary_zero_to_na"] = int(zero.sum())
        df.loc[zero, ["salary_minimum", "salary_maximum"]] = np.nan

        swapped = df["salary_minimum"] > df["salary_maximum"]
        audit["salary_swapped"] = int(swapped.sum())
        df.loc[swapped, ["salary_minimum", "salary_maximum"]] = (
            df.loc[swapped, ["salary_maximum", "salary_minimum"]].to_numpy()
        )

        impossible = (df["salary_minimum"] < SALARY_FLOOR) | (df["salary_maximum"] > SALARY_CEILING)
        audit["salary_impossible_to_na"] = int(impossible.sum())
        df.loc[impossible, ["salary_minimum", "salary_maximum"]] = np.nan

        midpoint = (df["salary_minimum"] + df["salary_maximum"]) / 2
        if "average_salary" in df:
            # Keep the source average where the bounds survived; blank it where they did not.
            df["average_salary"] = df["average_salary"].mask(df["average_salary"] <= 0)
            df["average_salary"] = df["average_salary"].where(midpoint.notna()).fillna(midpoint)
        else:
            df["average_salary"] = midpoint
    elif "average_salary" in df:
        df["average_salary"] = df["average_salary"].mask(df["average_salary"] <= 0)
        df["average_salary"] = df["average_salary"].where(
            df["average_salary"].between(SALARY_FLOOR, SALARY_CEILING)
        )
    else:
        df["average_salary"] = np.nan

    # Cap (do not delete) the thin top and bottom 1% of genuine salaries, so a few
    # executive packages cannot steer every average. Medians are unaffected.
    valid = df["average_salary"].dropna()
    if len(valid) >= 100:
        low, high = valid.quantile([0.01, 0.99])
        outside = (df["average_salary"] < low) | (df["average_salary"] > high)
        audit["salary_capped_1pct"] = int(outside.sum())
        df["average_salary"] = df["average_salary"].clip(low, high)

    # 5. Vacancies: a posting offers at least one seat.
    vacancies = _numeric(df, "numberOfVacancies")
    bad_vacancies = vacancies.isna() | (vacancies < 1)
    audit["vacancies_fixed"] = int(bad_vacancies.sum())
    df["numberOfVacancies"] = vacancies.where(~bad_vacancies, 1)

    # 6. Experience: blank implausible values rather than guess.
    years = _numeric(df, "minimumYearsExperience")
    too_many = years > MAX_EXPERIENCE_YEARS
    audit["experience_implausible_to_na"] = int(too_many.sum())
    df["minimumYearsExperience"] = years.mask(too_many)

    for column in ("metadata_totalNumberJobApplication", "metadata_totalNumberOfView", "metadata_repostCount"):
        if column in df:
            df[column] = _numeric(df, column)

    # 7. Text columns.
    for column in ("title", "employmentTypes", "positionLevels", "postedCompany_name"):
        if column in df:
            df[column] = df[column].astype("string").str.strip()
            df[column] = df[column].mask(df[column].isin(["", "nan"]))
    if "title" not in df:
        df["title"] = pd.Series("Untitled role", index=df.index, dtype="string")
    df["title_key"] = df["title"].str.strip().str.casefold()
    for column in ("employmentTypes", "positionLevels"):
        if column not in df:
            df[column] = pd.Series("Unspecified", index=df.index, dtype="string")

    df = add_features(df)
    audit["rows_out"] = len(df)
    return (df, audit) if return_audit else df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Columns that describe an opportunity rather than a raw posting."""
    df = df.copy()
    df["title_clean"] = clean_title(df["title"])
    df["title_clean_key"] = df["title_clean"].str.casefold()
    levels = df["positionLevels"].astype("string")
    df["seniority"] = pd.Categorical(levels.map(SENIORITY_BANDS), categories=SENIORITY_ORDER, ordered=True)
    df["pmet"] = np.where(levels.isin(NON_PMET_LEVELS), "Non-PMET", "PMET")
    df["pmet"] = df["pmet"].where(levels.notna(), "Unspecified")
    df["broad_sector"] = df["category"].map(CATEGORY_TO_SECTOR).fillna("Others")
    reposts = _numeric(df, "metadata_repostCount")
    df["is_repost"] = reposts.fillna(0) > 0
    years = _numeric(df, "minimumYearsExperience")
    df["is_entry_friendly"] = years <= 1
    first_posted = df["original_date"] if "original_date" in df else df["posted_date"]
    df["in_engagement_window"] = first_posted <= ENGAGEMENT_END
    return df


def clean_chunk(raw: pd.DataFrame):
    """Notebook-friendly name: clean a frame and return (cleaned, audit counts)."""
    return clean_jobs(raw, return_audit=True)


# ---------------------------------------------------------------------------
# Filtering (shared by every view in the app)
# ---------------------------------------------------------------------------

def filter_jobs(
    df: pd.DataFrame,
    sectors: Iterable[str] = (),
    categories: Iterable[str] = (),
    levels: Iterable[str] = (),
    employment: Iterable[str] = (),
    pmet: str = "All",
    exclude_reposts: bool = False,
    date_range: tuple | None = None,
) -> pd.DataFrame:
    """Apply sidebar filters. An empty selection means "no filter"."""
    mask = pd.Series(True, index=df.index)
    if sectors:
        mask &= df["broad_sector"].isin(list(sectors))
    if categories:
        mask &= df["category"].isin(list(categories))
    if levels:
        mask &= df["positionLevels"].isin(list(levels))
    if employment:
        mask &= df["employmentTypes"].isin(list(employment))
    if pmet != "All":
        mask &= df["pmet"] == pmet
    if exclude_reposts:
        mask &= ~df["is_repost"]
    if date_range is not None:
        start, end = (pd.Timestamp(d) for d in date_range)
        mask &= df["posted_date"].between(start, end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    return df.loc[mask]


def quality_summary(df: pd.DataFrame) -> dict[str, int]:
    """Small set of transparent data-quality counts for the dashboard."""
    check_columns = [c for c in df.columns if c != "category_list"]
    return {
        "rows": len(df),
        "duplicate_rows": int(df[check_columns].duplicated().sum()),
        "missing_titles": int(df["title"].isna().sum()),
        "missing_salaries": int(df["average_salary"].isna().sum()),
        "unparsed_dates": int(df["posted_date"].isna().sum()),
        "reposted_postings": int(df["is_repost"].sum()),
        "outside_engagement_window": int((~df["in_engagement_window"]).sum()),
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _one_row_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """A posting appears once for every category it belongs to."""
    if "category_list" in df:
        names = df["category_list"]
    elif "categories" in df:
        names = df["categories"].map(_category_names)
    else:
        names = df["category"].map(lambda value: [value] if isinstance(value, str) else [])
    long = df.assign(category_all=names).explode("category_all").dropna(subset=["category_all"])
    long["category_all"] = long["category_all"].astype(str)
    long["broad_sector_all"] = long["category_all"].map(CATEGORY_TO_SECTOR).fillna("Others")
    return long


def explode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook-friendly view: one row per posting per category, in a `category` column."""
    long = _one_row_per_category(df)
    return long.drop(columns=["category"]).rename(columns={"category_all": "category"})


def _engagement_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Applications, views and vacancies counted only inside the engagement window."""
    window = frame["in_engagement_window"] if "in_engagement_window" in frame else pd.Series(True, index=frame.index)
    return frame.assign(
        _apps=_numeric(frame, "metadata_totalNumberJobApplication").where(window),
        _views=_numeric(frame, "metadata_totalNumberOfView").where(window),
        _window_vacancies=_numeric(frame, "numberOfVacancies").where(window),
        _window_posting=window.astype(int),
    )


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)


def category_market(df: pd.DataFrame) -> pd.DataFrame:
    """One row per category: demand, pay, competition, interest and friction.

    Applications per vacancy and views per posting use only postings inside the
    engagement window; postings, vacancies and pay use every posting.
    """
    long = _one_row_per_category(df)
    if long.empty:
        return pd.DataFrame()
    long = _engagement_columns(long)
    grouped = long.groupby("category_all").agg(
        broad_sector=("broad_sector_all", "first"),
        postings=("title_key", "size"),
        vacancies=("numberOfVacancies", "sum"),
        median_salary=("average_salary", "median"),
        avg_experience=("minimumYearsExperience", "mean"),
        entry_friendly_share=("is_entry_friendly", "mean"),
        repost_share=("is_repost", "mean"),
        window_postings=("_window_posting", "sum"),
        applications=("_apps", "sum"),
        views=("_views", "sum"),
        window_vacancies=("_window_vacancies", "sum"),
    ).reset_index().rename(columns={"category_all": "category"})
    grouped["postings"] = grouped["postings"].astype("int64")  # counts of a text column come back as Int64
    grouped["apps_per_vacancy"] = _ratio(grouped["applications"], grouped["window_vacancies"])
    grouped["views_per_posting"] = _ratio(grouped["views"], grouped["window_postings"])
    grouped.loc[grouped["window_postings"] == 0, ["apps_per_vacancy", "views_per_posting"]] = np.nan
    return grouped.drop(columns=["applications", "views", "window_vacancies"])


def sector_summary(df: pd.DataFrame, top_titles: int = 3) -> pd.DataFrame:
    """Category table benchmarked against its broad sector (pay and experience)."""
    market = category_market(df)
    if market.empty:
        return market
    long = _one_row_per_category(df)
    benchmarks = long.groupby("broad_sector_all").agg(
        sector_median_salary=("average_salary", "median"),
        sector_avg_experience=("minimumYearsExperience", "mean"),
    )
    market = market.join(benchmarks, on="broad_sector")
    market["vacancy_share_pct"] = market["vacancies"] / market["vacancies"].sum() * 100
    market["salary_premium_pct"] = (market["median_salary"] / market["sector_median_salary"] - 1) * 100
    market["salary_vs_sector"] = np.select(
        [market["salary_premium_pct"] > 0, market["salary_premium_pct"] < 0],
        ["↑ Higher", "↓ Lower"], default="▬ Same",
    )
    gap = market["avg_experience"] - market["sector_avg_experience"]
    market["experience_vs_sector"] = np.select([gap > 0.2, gap < -0.2], ["↑ More", "↓ Less"], default="▬ Similar")
    market["repost_pct"] = market["repost_share"] * 100

    titles = (
        long.dropna(subset=["title_clean"])
        .groupby(["category_all", "title_clean_key"]).agg(name=("title_clean", "first"), n=("title_clean", "size"))
        .reset_index().sort_values(["category_all", "n"], ascending=[True, False])
        .groupby("category_all").head(top_titles)
        .groupby("category_all")["name"].agg(", ".join)
    )
    market["top_titles"] = market["category"].map(titles)
    return market.sort_values("vacancies", ascending=False).reset_index(drop=True)


def friction_premium(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """The top-N categories by vacancies: pay premium vs sector and % reposted."""
    table = sector_summary(df)
    return table.head(top_n) if not table.empty else table


def reward_barrier(df: pd.DataFrame, top_n: int = 10, min_postings: int = 500) -> pd.DataFrame:
    """The top-N most competitive categories with their pay (reward) and experience (barrier)."""
    market = category_market(df)
    if market.empty:
        return market
    market = market[(market["postings"] >= min_postings) & market["apps_per_vacancy"].notna()]
    return market.nlargest(top_n, "apps_per_vacancy").reset_index(drop=True)


PERIODS = {"Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}


def period_summary(df: pd.DataFrame, granularity: str = "Monthly") -> pd.DataFrame:
    """Vacancies per period split by PMET, with applications per vacancy where trustworthy."""
    dated = _engagement_columns(df.dropna(subset=["posted_date"]))
    if dated.empty:
        return pd.DataFrame()
    dated = dated.assign(period=dated["posted_date"].dt.to_period(PERIODS[granularity]).dt.start_time)
    grouped = dated.groupby(["period", "pmet"]).agg(
        postings=("title_key", "size"),
        vacancies=("numberOfVacancies", "sum"),
        applications=("_apps", "sum"),
        window_vacancies=("_window_vacancies", "sum"),
        window_postings=("_window_posting", "sum"),
    ).reset_index()
    grouped["postings"] = grouped["postings"].astype("int64")
    grouped["apps_per_vacancy"] = _ratio(grouped["applications"], grouped["window_vacancies"])
    # A period that is mostly outside the window would show a misleadingly low ratio.
    grouped.loc[grouped["window_postings"] < grouped["postings"] * 0.5, "apps_per_vacancy"] = np.nan
    return grouped.drop(columns=["applications", "window_vacancies"])


def skill_premiums(df: pd.DataFrame, min_postings: int = 500) -> pd.DataFrame:
    """Median salary of postings whose title mentions a keyword, against all postings."""
    overall = df["average_salary"].median()
    rows = []
    for skill, pattern in SKILL_KEYWORDS.items():
        mask = df["title_key"].str.contains(pattern, regex=True, na=False)
        salaries = df.loc[mask, "average_salary"].dropna()
        if mask.sum() < min_postings or salaries.empty:
            continue
        median = salaries.median()
        rows.append({
            "skill": skill,
            "postings": int(mask.sum()),
            "median_salary": median,
            "premium_pct": (median / overall - 1) * 100,
        })
    return pd.DataFrame(rows).sort_values("premium_pct", ascending=False) if rows else pd.DataFrame()


def career_tracks(df: pd.DataFrame, min_postings: int = 500, separator: str = " · ") -> pd.DataFrame:
    """Track = category + seniority band. Scores are percentile ranks (0-100)."""
    frame = _one_row_per_category(df)
    frame = frame.assign(seniority_band=frame["positionLevels"].map(SENIORITY_BANDS)).dropna(subset=["seniority_band"])
    if frame.empty:
        return pd.DataFrame()
    frame = _engagement_columns(frame)
    frame = frame.assign(
        track=frame["category_all"] + separator + frame["seniority_band"].astype(str),
        years=_numeric(frame, "minimumYearsExperience"),
    )
    grouped = frame.groupby("track").agg(
        postings=("title_key", "size"),
        median_salary=("average_salary", "median"),
        applications=("_apps", "sum"),
        window_vacancies=("_window_vacancies", "sum"),
        accessible_share=("years", lambda s: (s <= 3).mean()),
    ).reset_index()
    grouped["postings"] = grouped["postings"].astype("int64")
    grouped = grouped[grouped["postings"] >= min_postings].copy()
    if grouped.empty:
        return grouped
    grouped["apps_per_vacancy"] = _ratio(grouped["applications"], grouped["window_vacancies"])
    grouped["demand_score"] = grouped["postings"].rank(pct=True) * 100
    grouped["pay_score"] = grouped["median_salary"].rank(pct=True) * 100
    # Tracks with no in-window postings get a neutral 50 rather than looking uncontested.
    grouped["low_competition_score"] = ((-grouped["apps_per_vacancy"]).rank(pct=True) * 100).fillna(50)
    grouped["access_score"] = grouped["accessible_share"].rank(pct=True) * 100
    return grouped.drop(columns=["applications", "window_vacancies"])


def fit_score(tracks: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted average of the four percentile scores, using slider weights."""
    total = sum(weights.values()) or 1.0
    return (
        tracks["demand_score"] * weights["demand"]
        + tracks["pay_score"] * weights["pay"]
        + tracks["low_competition_score"] * weights["competition"]
        + tracks["access_score"] * weights["access"]
    ) / total


def monthly_postings(df: pd.DataFrame, by_category: bool = False) -> pd.DataFrame:
    """Postings per month from the selected posting date."""
    dated = df.dropna(subset=["posted_date"]).copy()
    dated["month"] = dated["posted_date"].dt.to_period("M").dt.to_timestamp()
    keys = ["month", "category"] if by_category else ["month"]
    return dated.groupby(keys).size().rename("Postings").reset_index()


# ---------------------------------------------------------------------------
# Notebook: build every analysis table from the raw CSV in one call
# ---------------------------------------------------------------------------

def _growth_6m(by_cat: pd.DataFrame) -> pd.Series:
    """% change in postings: last 6 complete months vs the 6 before (from May 2023)."""
    dated = by_cat[by_cat["posted_date"] >= TREND_CUTOFF].dropna(subset=["posted_date"])
    if dated.empty:
        return pd.Series(dtype="float64")
    months = dated["posted_date"].dt.to_period("M")
    last = months.max()
    recent = dated[months > last - 6]
    previous = dated[(months <= last - 6) & (months > last - 12)]
    now = recent.groupby("category").size()
    before = previous.groupby("category").size()
    return ((now / before.where(before > 0)) - 1) * 100


def build_analysis_tables(raw: pd.DataFrame, track_min_postings: int = 500):
    """Return (jobs, by_cat, categories, skills, tracks, quality) for the notebook."""
    cleaned, audit = clean_jobs(raw, return_audit=True)

    jobs = cleaned.rename(columns={
        "numberOfVacancies": "vacancies",
        "postedCompany_name": "company",
        "average_salary": "avg_salary",
    }).assign(is_hard_to_fill=cleaned["is_repost"])

    by_cat = explode_categories(cleaned)

    categories = category_market(cleaned).rename(columns={"repost_share": "hard_to_fill_share"})
    categories["growth_pct_6m"] = categories["category"].map(_growth_6m(by_cat))
    categories = categories.sort_values("postings", ascending=False).reset_index(drop=True)

    skills = skill_premiums(cleaned, min_postings=1).rename(columns={"premium_pct": "salary_premium_pct"})

    tracks = career_tracks(cleaned, min_postings=track_min_postings, separator=" - ")
    if not tracks.empty:
        tracks["accessibility_score_default"] = tracks["access_score"]
        default = {"demand": 30, "pay": 30, "competition": 25, "access": 15}
        tracks["fit_score_default"] = fit_score(tracks, default)
        tracks = tracks.sort_values("fit_score_default", ascending=False).reset_index(drop=True)

    quality = {"rows_read": audit["rows_in"], "rows_after_cleaning": audit["rows_out"], **audit}
    return jobs, by_cat, categories, skills, tracks, quality
