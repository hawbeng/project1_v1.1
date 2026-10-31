# Singapore Jobs Explorer: a Streamlit dashboard for career coaches and job seekers.
# Run with:  streamlit run app.py   (needs data/SGJobData.csv next to this file)

# Standard library: Path helps build file locations relative to this script.
from pathlib import Path

# numpy and pandas: number and table libraries.
import numpy as np
import pandas as pd

# plotly: interactive charts. express for quick charts, graph_objects for combined ones.
# pyrefly: ignore[missing-import]
import plotly.express as px
# pyrefly: ignore[missing-import]
import plotly.graph_objects as go
# pyrefly: ignore[missing-import]
from plotly.subplots import make_subplots

# streamlit: the web app framework that builds the page from Python code.
# pyrefly: ignore[missing-import]
import streamlit as st

# Helper functions and constants from data_utils.py (our own module).
from data_utils import (
    BROAD_SECTORS,
    ENGAGEMENT_END,
    PERIODS,
    SECTOR_COLOURS,
    SECTOR_ORDER,
    TREND_CUTOFF,
    career_tracks,
    category_market,
    clean_jobs,
    filter_jobs,
    fit_score,
    friction_premium,
    monthly_postings,
    period_summary,
    quality_summary,
    read_jobs,
    reward_barrier,
    sector_summary,
    skill_premiums,
)


# Folder that contains this app.py file, and the course dataset inside it.
ROOT = Path(__file__).parent
COURSE_FILE = ROOT / "data" / "SGJobData.csv"

# Streamlit 1.46+ sizes elements with width="stretch" and shows "%,.0f" with thousands
# separators. Older versions crash on a text width and flag "%,.0f" as invalid, so fall back.
NEW_STREAMLIT = tuple(int(p) for p in st.__version__.split(".")[:2]) >= (1, 46)
STRETCH = {"width": "stretch"} if NEW_STREAMLIT else {"use_container_width": True}
MONEY_FORMAT = "%,.0f" if NEW_STREAMLIT else "%.0f"

# Label for the end of the engagement window, used in captions.
WINDOW_LABEL = ENGAGEMENT_END.strftime("%B %Y")

# Browser tab title, icon, and full-width layout.
st.set_page_config(page_title="Singapore Jobs Explorer", page_icon="📊", layout="wide")
st.title("Singapore Jobs Explorer")
st.caption("A practical view of role demand, pay and competition for career conversations.")


# ---------------------------------------------------------------------------
# 1. LOAD ONCE, THEN REUSE
# ---------------------------------------------------------------------------
# Reading and cleaning a million rows takes a few seconds, so we do it once per
# server process. cache_resource keeps the cleaned table in memory and hands the
# same object to every rerun (a slider move no longer re-reads the CSV).
@st.cache_resource(show_spinner="Loading and cleaning the job postings (first run only)...")
def load_jobs(path: str, modified: float):
    raw = read_jobs(path)
    return clean_jobs(raw, return_audit=True)


# Stop with a clear message, rather than a traceback, if the course file is missing.
if not COURSE_FILE.exists():
    st.error(
        "Could not find **data/SGJobData.csv**. Copy the course file into the `data/` "
        "folder next to app.py, then refresh this page."
    )
    st.stop()

df, cleaning_log = load_jobs(str(COURSE_FILE), COURSE_FILE.stat().st_mtime)


# ---------------------------------------------------------------------------
# 2. SIDEBAR FILTERS
# ---------------------------------------------------------------------------
sector_help = "**Broad sectors group the 43 categories:**\n\n" + "\n".join(
    f"- **{sector}**: {', '.join(cats)}" for sector, cats in BROAD_SECTORS.items()
)

with st.sidebar:
    st.subheader("Filters")
    st.caption("An empty filter means **all**.")

    # Broad sector first, so the category list can be narrowed to match it.
    selected_sectors = st.multiselect(
        "Broad sector", [s for s in SECTOR_ORDER if s in set(df["broad_sector"])], help=sector_help
    )
    sector_rows = df[df["broad_sector"].isin(selected_sectors)] if selected_sectors else df
    selected_categories = st.multiselect(
        "Industry / category", sorted(sector_rows["category"].dropna().unique())
    )
    selected_levels = st.multiselect("Career level", sorted(df["positionLevels"].dropna().unique()))
    selected_employment = st.multiselect("Employment type", sorted(df["employmentTypes"].dropna().unique()))

    selected_pmet = st.radio(
        "PMET status", ["All", "PMET", "Non-PMET"], horizontal=True,
        help="PMET = professionals, managers, executives and technicians. "
             "Rule used here: only 'Non-executive' postings are Non-PMET.",
    )
    exclude_reposts = st.checkbox(
        "Exclude reposted adverts", help="A repost is an advert put up again, often because it was hard to fill."
    )

    first_day, last_day = df["posted_date"].min().date(), df["posted_date"].max().date()
    picked = st.date_input(
        "Posting date range", value=(first_day, last_day), min_value=first_day, max_value=last_day
    )
    # While the user is picking the second date, date_input returns only one date.
    date_range = tuple(picked) if isinstance(picked, (list, tuple)) and len(picked) == 2 else (first_day, last_day)

# Every filter as one hashable key, so cached results can be reused per combination.
FILTER_KEY = (
    COURSE_FILE.stat().st_mtime,
    tuple(selected_sectors), tuple(selected_categories), tuple(selected_levels),
    tuple(selected_employment), selected_pmet, exclude_reposts, date_range,
)


@st.cache_resource(max_entries=12, show_spinner=False)
def get_filtered(key) -> pd.DataFrame:
    """The filtered rows. Reused (not copied) on reruns, so never modify the result."""
    _, sectors, categories, levels, employment, pmet, reposts, dates = key
    return filter_jobs(df, sectors, categories, levels, employment, pmet, reposts, dates)


# Small summary tables are cheap to copy, so cache_data is the right tool for them.
@st.cache_data(show_spinner=False)
def get_market(key):
    return category_market(get_filtered(key))


@st.cache_data(show_spinner=False)
def get_sector_table(key):
    return sector_summary(get_filtered(key))


@st.cache_data(show_spinner=False)
def get_periods(key, granularity):
    return period_summary(get_filtered(key), granularity)


@st.cache_data(show_spinner=False)
def get_tracks(key, min_postings):
    return career_tracks(get_filtered(key), min_postings)


@st.cache_data(show_spinner=False)
def get_skills(key, min_postings):
    return skill_premiums(get_filtered(key), min_postings)


filtered = get_filtered(FILTER_KEY)
st.info(f"Showing **full course dataset** · {len(filtered):,} of {len(df):,} postings after filters")
if filtered.empty:
    st.warning("No postings match these filters. Widen the filters or the date range.")
    st.stop()


# ---------------------------------------------------------------------------
# 3. HEADLINE NUMBERS
# ---------------------------------------------------------------------------
salary = filtered["average_salary"].dropna()
window = filtered[filtered["in_engagement_window"]]
window_vacancies = window["numberOfVacancies"].sum()
apps_per_vacancy = window["metadata_totalNumberJobApplication"].sum() / window_vacancies if window_vacancies else None

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Job postings", f"{len(filtered):,}")
m2.metric("Vacancies", f"{int(filtered['numberOfVacancies'].sum()):,}", help="Open seats. One posting can offer several.")
m3.metric("Median listed salary", f"S${salary.median():,.0f}" if not salary.empty else "n/a")
m4.metric("Salary coverage", f"{filtered['average_salary'].notna().mean():.0%}", help="Share of postings that list a valid salary.")
m5.metric("Reposted adverts", f"{filtered['is_repost'].mean():.1%}", help="Share of postings put up more than once.")
m6.metric(
    "Applications per vacancy", f"{apps_per_vacancy:.2f}" if apps_per_vacancy is not None else "n/a",
    help=f"Uses adverts first posted up to {WINDOW_LABEL} only. Application counts stop "
         "building up after that, so later adverts would make competition look far lower than it is.",
)


def show(fig, height: int | None = None):
    """Draw a Plotly figure at full width with a consistent look."""
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, **STRETCH)


(
    overview, salary_tab, trends_tab, demand_tab, friction_tab,
    skills_tab, career_tab, table_tab, quality_tab,
) = st.tabs([
    "Overview", "Salary", "Trends", "Demand vs competition", "Pay vs friction",
    "Skills", "Career recommender", "Sector table", "Data quality",
])


# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------
with overview:
    left, right = st.columns(2)

    # Count postings per cleaned title, so "Urgent Hiring!!! Sales Executive" and
    # "Sales Executive" are one role.
    titles = (
        filtered.dropna(subset=["title_clean_key"])
        .groupby("title_clean_key", as_index=False)
        .agg(Role=("title_clean", "first"), Postings=("title_clean", "size"))
        .nlargest(12, "Postings")
    )
    left.subheader("Most-posted job titles")
    left.plotly_chart(
        px.bar(titles.sort_values("Postings"), x="Postings", y="Role", orientation="h",
               color_discrete_sequence=["#2F6B5F"]),
        **STRETCH,
    )

    category_counts = (
        filtered.dropna(subset=["category"])
        .groupby(["category", "broad_sector"], as_index=False).size()
        .rename(columns={"size": "Postings"}).nlargest(10, "Postings")
    )
    right.subheader("Posting volume by category")
    right.plotly_chart(
        px.bar(category_counts.sort_values("Postings"), x="Postings", y="category", orientation="h",
               color="broad_sector", color_discrete_map=SECTOR_COLOURS,
               labels={"category": "Category", "broad_sector": "Broad sector"})
        .update_layout(legend=dict(orientation="h", y=-0.25, title_text="")),
        **STRETCH,
    )

    st.subheader("Vacancies by broad sector")
    by_sector = (
        filtered.groupby("broad_sector", as_index=False)["numberOfVacancies"].sum()
        .sort_values("numberOfVacancies")
    )
    show(px.bar(by_sector, x="numberOfVacancies", y="broad_sector", orientation="h",
                color="broad_sector", color_discrete_map=SECTOR_COLOURS,
                labels={"numberOfVacancies": "Vacancies", "broad_sector": ""}).update_layout(showlegend=False),
         height=320)
    st.caption(
        "Titles are cleaned of hiring slogans, bracketed detail and pay phrases. Category and sector "
        "charts use each posting's first-listed category. These describe posting volume, not unique "
        "employers or actual hires."
    )


# ---------------------------------------------------------------------------
# SALARY
# ---------------------------------------------------------------------------
with salary_tab:
    salary_rows = filtered.dropna(subset=["average_salary"])
    if salary_rows.empty:
        st.warning("No valid salary values are available for the current selection.")
    else:
        left, right = st.columns(2)

        left.subheader("Pay ladder by seniority")
        ladder = (
            salary_rows.dropna(subset=["seniority"])
            .groupby("seniority", observed=True)["average_salary"]
            .agg(median="median", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75))
            .reset_index()
        )
        fig = px.bar(ladder, x="seniority", y="median", color_discrete_sequence=["#2F6FB0"],
                     labels={"seniority": "Seniority band", "median": "Median salary (S$)"})
        fig.update_traces(error_y=dict(type="data", symmetric=False,
                                       array=ladder["p75"] - ladder["median"],
                                       arrayminus=ladder["median"] - ladder["p25"]))
        left.plotly_chart(fig, **STRETCH)
        left.caption("Bars are medians; whiskers show the middle half of listed salaries (25th–75th percentile).")

        right.subheader("Median listed salary by category")
        salary_by_category = (
            salary_rows.groupby(["category", "broad_sector"], as_index=False)["average_salary"].median()
            .nlargest(12, "average_salary")
        )
        right.plotly_chart(
            px.bar(salary_by_category.sort_values("average_salary"), x="average_salary", y="category",
                   orientation="h", color="broad_sector", color_discrete_map=SECTOR_COLOURS,
                   labels={"average_salary": "Median salary (S$)", "category": "", "broad_sector": "Broad sector"})
            .update_layout(legend=dict(orientation="h", y=-0.25, title_text="")),
            **STRETCH,
        )
        st.caption(
            "Salary is the advertised monthly midpoint. Zero salaries are 'not disclosed'; ranges entered "
            "the wrong way round are swapped; values below S\\$500 or above S\\$60,000 a month are left out."
        )


# ---------------------------------------------------------------------------
# TRENDS
# ---------------------------------------------------------------------------
with trends_tab:
    c1, c2 = st.columns([2, 3])
    granularity = c1.radio("Time unit", list(PERIODS), index=1, horizontal=True)
    show_early = c2.checkbox(
        f"Include periods before {TREND_CUTOFF:%B %Y} (collection was incomplete; not comparable)", value=False
    )

    periods = get_periods(FILTER_KEY, granularity)
    if periods.empty:
        st.warning("No dates could be read for the current selection.")
    else:
        if not show_early:
            periods = periods[periods["period"] >= TREND_CUTOFF.to_period(PERIODS[granularity]).start_time]

        st.subheader(f"Vacancies and competition ({granularity.lower()})")
        combo = make_subplots(specs=[[{"secondary_y": True}]])
        for status, colour in (("PMET", "#2F6FB0"), ("Non-PMET", "#E0B07A")):
            part = periods[periods["pmet"] == status]
            combo.add_trace(go.Bar(x=part["period"], y=part["vacancies"], name=f"Vacancies ({status})",
                                   marker_color=colour), secondary_y=False)
        for status, colour in (("PMET", "#14213D"), ("Non-PMET", "#A54A12")):
            part = periods[(periods["pmet"] == status)].dropna(subset=["apps_per_vacancy"])
            if not part.empty:
                combo.add_trace(go.Scatter(x=part["period"], y=part["apps_per_vacancy"].round(2),
                                           name=f"Applications per vacancy ({status})", mode="lines+markers",
                                           line=dict(color=colour, width=3)), secondary_y=True)
        combo.update_layout(barmode="stack", hovermode="x unified")
        combo.update_yaxes(title_text="Vacancies", secondary_y=False)
        combo.update_yaxes(title_text="Applications per vacancy", secondary_y=True, showgrid=False, rangemode="tozero")
        show(combo, height=460)
        st.caption(
            f"Bars use the latest posting date. The competition line appears only for periods where most "
            f"adverts were first posted by {WINDOW_LABEL}; after that, application counts stop building up. "
            "The later fall in volume is visible, but its cause is not confirmed."
        )

        st.subheader("Compare categories over time")
        dated = filtered.dropna(subset=["posted_date"])
        top_categories = dated["category"].value_counts().head(5).index.tolist()
        chosen = st.multiselect("Categories", sorted(dated["category"].dropna().unique()), default=top_categories)
        if chosen:
            by_category = monthly_postings(dated[dated["category"].isin(chosen)], by_category=True)
            if not show_early:
                by_category = by_category[by_category["month"] >= TREND_CUTOFF]
            show(px.line(by_category, x="month", y="Postings", color="category", markers=True))


# ---------------------------------------------------------------------------
# DEMAND VS COMPETITION
# ---------------------------------------------------------------------------
with demand_tab:
    market = get_market(FILTER_KEY)
    if market.empty or market["apps_per_vacancy"].notna().sum() == 0:
        st.info(f"No adverts in this selection were first posted by {WINDOW_LABEL}, so competition cannot be measured.")
    else:
        min_postings = st.slider("Minimum postings per category", 500, 20_000, 2_000, step=500)
        view = market[market["postings"] >= min_postings].dropna(subset=["median_salary", "apps_per_vacancy"]).copy()
        if view.empty:
            st.warning("No categories meet this minimum. Lower the slider.")
        else:
            x_split, y_split = view["apps_per_vacancy"].median(), view["postings"].median()
            view["Quadrant"] = np.select(
                [
                    (view["apps_per_vacancy"] < x_split) & (view["postings"] >= y_split),
                    (view["apps_per_vacancy"] < x_split) & (view["postings"] < y_split),
                    (view["apps_per_vacancy"] >= x_split) & (view["postings"] >= y_split),
                ],
                ["Big and open", "Small and open", "Big and crowded"],
                default="Small and crowded",
            )

            st.subheader("Demand (postings) against competition (applications per vacancy)")
            fig = px.scatter(
                view, x="apps_per_vacancy", y="postings", color="Quadrant", size="median_salary",
                hover_name="category", log_y=True,
                labels={"apps_per_vacancy": f"Applications per vacancy (adverts to {WINDOW_LABEL})",
                        "postings": "Postings (log scale)"},
                color_discrete_map={"Big and open": "#2F6B5F", "Small and open": "#8FB8A8",
                                    "Big and crowded": "#D18B47", "Small and crowded": "#B8C4CC"},
            )
            fig.add_vline(x=x_split, line_dash="dot", line_color="#8B95A1")
            fig.add_hline(y=y_split, line_dash="dot", line_color="#8B95A1")
            show(fig, height=500)
            st.caption(
                f"Dotted lines are the medians of the categories shown: {x_split:.1f} applications per vacancy "
                f"and {y_split:,.0f} postings. Bubble size is median pay. A posting counts in every category it "
                "belongs to. Applications per vacancy = total applications ÷ total vacancies."
            )

            # Interest vs supply: how much attention each category's adverts get.
            st.subheader("Job-seeker interest against supply")
            view["views_per_posting"] = view["views_per_posting"].round(1)
            top_three = view.nlargest(3, "apps_per_vacancy")["category"].tolist()
            avg_views = (
                window["metadata_totalNumberOfView"].sum() / len(window) if len(window) else np.nan
            )
            bubble = px.scatter(
                view, x="vacancies", y="views_per_posting", size="apps_per_vacancy", color="broad_sector",
                color_discrete_map=SECTOR_COLOURS, hover_name="category", log_x=True, size_max=40,
                labels={"vacancies": "Vacancies (log scale)",
                        "views_per_posting": f"Views per posting (adverts to {WINDOW_LABEL})",
                        "broad_sector": "Broad sector"},
            )
            for trace in bubble.data:
                names = list(trace.hovertext)
                trace.marker.line.width = [3 if n in top_three else 0.5 for n in names]
                trace.marker.line.color = ["#C0392B" if n in top_three else "#FFFFFF" for n in names]
            if pd.notna(avg_views):
                bubble.add_hline(y=avg_views, line_dash="dot", line_color="#8B95A1",
                                 annotation_text=f"Average views per posting: {avg_views:.0f}",
                                 annotation_position="bottom left")
            bubble.add_vline(x=view["vacancies"].median(), line_dash="dot", line_color="#8B95A1")
            for x, y, xa, ya, text in (
                (0.99, 0.99, "right", "top", "Hotspot: many seats, much interest"),
                (0.01, 0.99, "left", "top", "Niche: few seats, much interest"),
                (0.99, 0.01, "right", "bottom", "Opportunity: many seats, less interest"),
                (0.01, 0.01, "left", "bottom", "Quiet zone: few seats, less interest"),
            ):
                bubble.add_annotation(x=x, y=y, xref="paper", yref="paper", xanchor=xa, yanchor=ya,
                                      text=text, showarrow=False, font=dict(size=12, color="#4A5568"),
                                      bgcolor="rgba(255,255,255,0.7)")
            show(bubble, height=520)
            st.caption(
                "Bubble size is applications per vacancy; red outlines mark the three most competitive "
                "categories shown. The quadrant names are prompts for a conversation, not conclusions."
            )

            st.dataframe(
                view.sort_values("apps_per_vacancy")[
                    ["category", "broad_sector", "postings", "vacancies", "median_salary",
                     "apps_per_vacancy", "views_per_posting", "entry_friendly_share", "Quadrant"]
                ].rename(columns={
                    "category": "Category", "broad_sector": "Broad sector", "postings": "Postings",
                    "vacancies": "Vacancies", "median_salary": "Median salary (S$)",
                    "apps_per_vacancy": "Applications per vacancy", "views_per_posting": "Views per posting",
                    "entry_friendly_share": "Entry-friendly share",
                }).round({"Median salary (S$)": 0, "Applications per vacancy": 2, "Entry-friendly share": 2}),
                hide_index=True, **STRETCH,
            )


# ---------------------------------------------------------------------------
# PAY VS FRICTION
# ---------------------------------------------------------------------------
with friction_tab:
    st.subheader("Reward against barrier in the most competitive categories")
    c1, c2 = st.columns(2)
    top_n = c1.slider("Categories to show", 5, 15, 10)
    barrier_min = c2.slider("Minimum postings to be ranked", 100, 10_000, 500, step=100)
    rb = reward_barrier(filtered, top_n=top_n, min_postings=barrier_min)
    if rb.empty:
        st.info(f"Not enough adverts first posted by {WINDOW_LABEL} to rank competition. Lower the minimum.")
    else:
        rb = rb.sort_values("apps_per_vacancy")
        labels = rb["category"] + " (" + rb["apps_per_vacancy"].round(1).astype(str) + ")"
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                            subplot_titles=("Reward: median salary (S$)", "Barrier: average years of experience asked"))
        fig.add_trace(go.Bar(y=labels, x=rb["median_salary"], orientation="h", marker_color="#2F6FB0",
                             text=rb["median_salary"].map("S${:,.0f}".format), textposition="outside",
                             cliponaxis=False), row=1, col=1)
        fig.add_trace(go.Bar(y=labels, x=rb["avg_experience"], orientation="h", marker_color="#E07A2E",
                             text=rb["avg_experience"].map("{:.1f} yrs".format), textposition="outside",
                             cliponaxis=False), row=1, col=2)
        # Fixed ranges leave room for the value labels; the reversed left axis mirrors the right one.
        fig.update_xaxes(range=[rb["median_salary"].max() * 1.35, 0], row=1, col=1)
        fig.update_xaxes(range=[0, rb["avg_experience"].max() * 1.3], row=1, col=2)
        fig.update_layout(showlegend=False)
        show(fig, height=max(360, 42 * len(rb)))
        st.caption(
            "Ranked by applications per vacancy (in brackets), most competitive at the top. A crowded category "
            "that also asks for little experience is easy to enter but hard to win."
        )

    st.subheader("Friction against pay premium in the highest-volume categories")
    fp_n = st.slider("Highest-volume categories to show", 5, 20, 10)
    fp = friction_premium(filtered, top_n=fp_n)
    if fp.empty:
        st.info("No categories in this selection.")
    else:
        fp = fp.assign(size=fp["apps_per_vacancy"].fillna(fp["apps_per_vacancy"].median()).fillna(1).clip(lower=0.1))
        friction_mid = fp["repost_pct"].median()
        fig = px.scatter(
            fp, x="salary_premium_pct", y="repost_pct", size="size", color="median_salary",
            text="category", hover_name="category", color_continuous_scale="Blues", size_max=45,
            hover_data={"size": False, "apps_per_vacancy": ":.2f", "vacancies": ":,", "median_salary": ":,.0f"},
            labels={"salary_premium_pct": "Pay premium vs its broad sector's median (%)",
                    "repost_pct": "Friction: share of adverts reposted (%)", "median_salary": "Median S$",
                    "apps_per_vacancy": "Applications per vacancy"},
        )
        fig.update_traces(textposition="top center", marker=dict(opacity=0.85, line=dict(width=1, color="white")))
        fig.add_vline(x=0, line_dash="dash", line_color="#8B95A1")
        fig.add_hline(y=friction_mid, line_dash="dash", line_color="#8B95A1")
        for x, y, xa, ya, text in (
            (0.99, 0.99, "right", "top", "Scarce talent? Pays more, reposted more"),
            (0.01, 0.99, "left", "top", "Hard to keep? Pays less, reposted more"),
            (0.99, 0.01, "right", "bottom", "Stable premium: pays more, reposted less"),
            (0.01, 0.01, "left", "bottom", "Fluid supply: pays less, reposted less"),
        ):
            fig.add_annotation(x=x, y=y, xref="paper", yref="paper", xanchor=xa, yanchor=ya, text=text,
                               showarrow=False, font=dict(size=12, color="#4A5568"), bgcolor="rgba(255,255,255,0.7)")
        show(fig, height=560)
        st.caption(
            "Pay premium compares a category's median salary with the median of its broad sector. Friction is the "
            "share of adverts put up more than once, a hint that roles are hard to fill. Bubble size is applications "
            f"per vacancy (adverts to {WINDOW_LABEL}). The corner labels are questions to ask, not findings."
        )


# ---------------------------------------------------------------------------
# SKILLS
# ---------------------------------------------------------------------------
with skills_tab:
    min_skill_postings = st.slider("Minimum postings per keyword", 100, 5_000, 500, step=100)
    skills = get_skills(FILTER_KEY, min_skill_postings)
    if skills.empty:
        st.warning("No keyword has enough postings. Lower the slider.")
    else:
        st.subheader("Salary premium for job titles mentioning each keyword")
        ordered = skills.sort_values("premium_pct")
        fig = px.bar(
            ordered, x="premium_pct", y="skill", orientation="h",
            labels={"premium_pct": "Premium vs all postings (%)", "skill": "Keyword in job title"},
            color=np.where(ordered["premium_pct"] >= 0, "Above overall median", "Below overall median"),
            color_discrete_map={"Above overall median": "#2F6B5F", "Below overall median": "#C0392B"},
        )
        show(fig)
        st.caption(
            "Skills are detected from job titles only, using the keyword list in data_utils.py. "
            "Premium = median salary of postings with the keyword ÷ median salary of all postings − 1. "
            "It shows association, not the effect of learning the skill."
        )
        st.dataframe(
            skills.rename(columns={"skill": "Keyword", "postings": "Postings", "median_salary": "Median salary (S$)",
                                   "premium_pct": "Premium (%)"}).round({"Median salary (S$)": 0, "Premium (%)": 1}),
            hide_index=True, **STRETCH,
        )


# ---------------------------------------------------------------------------
# CAREER RECOMMENDER
# ---------------------------------------------------------------------------
with career_tab:
    st.subheader("Find career tracks that fit what matters to you")
    st.caption(
        "A track is one category and one seniority band. Each factor is scored 0–100 across all tracks, "
        "then combined using your weights."
    )
    w_demand, w_pay, w_competition, w_access = st.columns(4)
    weights = {
        "demand": w_demand.slider("Demand", 0, 5, 3),
        "pay": w_pay.slider("Pay", 0, 5, 3),
        "competition": w_competition.slider("Low competition", 0, 5, 2),
        "access": w_access.slider("Easy to enter", 0, 5, 2),
    }
    min_track_postings = st.slider("Minimum postings per track", 200, 5_000, 1_000, step=100)
    tracks = get_tracks(FILTER_KEY, min_track_postings)
    if tracks.empty:
        st.warning("No career tracks meet this minimum. Lower the slider.")
    else:
        tracks = tracks.assign(fit=fit_score(tracks, weights)).sort_values("fit", ascending=False).head(15)
        st.dataframe(
            tracks.assign(**{"Needs ≤3 years (%)": tracks["accessible_share"] * 100})[
                ["track", "fit", "postings", "median_salary", "apps_per_vacancy", "Needs ≤3 years (%)"]
            ].rename(columns={
                "track": "Career track", "fit": "Fit score", "postings": "Postings",
                "median_salary": "Median salary (S$)", "apps_per_vacancy": "Applications per vacancy",
            }).round({"Fit score": 0, "Median salary (S$)": 0, "Applications per vacancy": 2, "Needs ≤3 years (%)": 0}),
            hide_index=True, **STRETCH,
        )
        st.caption(
            f"Competition uses adverts first posted by {WINDOW_LABEL}; a track with none scores a neutral 50. "
            "This ranks tracks by the weights you set. It is a guide for exploration, not a prediction of hiring or success."
        )


# ---------------------------------------------------------------------------
# SECTOR TABLE
# ---------------------------------------------------------------------------
with table_tab:
    st.subheader("Every category, benchmarked against its broad sector")
    table = get_sector_table(FILTER_KEY)
    if table.empty:
        st.info("No categories in this selection.")
    else:
        quick = st.radio(
            "Show", ["All categories", "Top 10 most competitive", "Top 10 by vacancies"], horizontal=True
        )
        if quick == "Top 10 most competitive":
            table = table.dropna(subset=["apps_per_vacancy"]).nlargest(10, "apps_per_vacancy")
        elif quick == "Top 10 by vacancies":
            table = table.nlargest(10, "vacancies")

        columns = {
            "category": "Category", "broad_sector": "Broad sector", "postings": "Postings",
            "vacancies": "Vacancies", "vacancy_share_pct": "% of vacancies",
            "apps_per_vacancy": "Applications per vacancy", "median_salary": "Median salary (S$)",
            "salary_vs_sector": "Pay vs sector", "sector_median_salary": "Sector median (S$)",
            "avg_experience": "Avg years asked", "experience_vs_sector": "Experience vs sector",
            "repost_pct": "% reposted", "views_per_posting": "Views per posting",
            "top_titles": "Most common titles",
        }
        display = table[list(columns)].rename(columns=columns)
        st.dataframe(
            display, hide_index=True, **STRETCH,
            column_config={
                "% of vacancies": st.column_config.ProgressColumn(
                    "% of vacancies", format="%.1f%%", min_value=0.0,
                    max_value=float(max(display["% of vacancies"].max(), 1.0))),
                "Applications per vacancy": st.column_config.NumberColumn(format="%.2f"),
                "Median salary (S$)": st.column_config.NumberColumn(format=MONEY_FORMAT),
                "Sector median (S$)": st.column_config.NumberColumn(format=MONEY_FORMAT),
                "Avg years asked": st.column_config.NumberColumn(format="%.1f"),
                "% reposted": st.column_config.NumberColumn(format="%.1f"),
                "Views per posting": st.column_config.NumberColumn(format="%.0f"),
                "Most common titles": st.column_config.TextColumn(width="large"),
            },
        )
        st.download_button(
            "Download this table (CSV)", display.to_csv(index=False).encode("utf-8"),
            file_name="sector_table.csv", mime="text/csv",
        )
        st.caption(
            "A posting counts in every category it belongs to, so vacancy shares overlap. Sector benchmarks are "
            f"medians and means across the whole broad sector. Applications and views use adverts first posted by {WINDOW_LABEL}."
        )


# ---------------------------------------------------------------------------
# DATA QUALITY
# ---------------------------------------------------------------------------
with quality_tab:
    left, right = st.columns(2)
    left.subheader("Checks on the selected data")
    quality = quality_summary(filtered)
    left.dataframe(
        pd.DataFrame([{"Check": k.replace("_", " ").capitalize(), "Count": v} for k, v in quality.items()]),
        hide_index=True, **STRETCH,
    )
    right.subheader("What cleaning changed (whole file)")
    right.dataframe(
        pd.DataFrame([{"Rule": k.replace("_", " ").capitalize(), "Rows": v} for k, v in cleaning_log.items()]),
        hide_index=True, **STRETCH,
    )
    st.write(
        "Missing salary values remain missing. Dates that cannot be parsed are excluded from the time trend. "
        f"Views and applications are only used for adverts first posted by {WINDOW_LABEL}. "
        "Review these counts before using the charts in a decision."
    )
