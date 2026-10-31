# Singapore Jobs Explorer

A Streamlit dashboard for career coaches and job seekers to explore Singapore job postings, compare pay, and spot roles worth investigating. The project follows the assignment's data workflow: load, clean, explore, and communicate.

## Start the dashboard

Copy the instructor-provided `SGJobData.csv` into `data/` (the folder is git-ignored), then:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first load reads and cleans all ~1M rows (a few seconds to a minute, depending on the machine). The cleaned table is then cached, so moving a filter or slider does not re-read the file.

## What the app shows

Sidebar filters (an empty filter means "all"): **broad sector**, industry/category, career level, employment type, **PMET status**, **exclude reposted adverts**, and a **posting date range**. Every tab uses the same filtered rows.

Headline numbers: postings, vacancies, median listed salary, salary coverage, share of reposted adverts, and applications per vacancy.

| Tab | What it answers |
|---|---|
| Overview | Most-posted (cleaned) job titles, biggest categories, vacancies by broad sector |
| Salary | Pay ladder by seniority (median and middle half), best-paid categories |
| Trends | Vacancies per week/month/quarter/year split by PMET, with competition where it can be measured |
| Demand vs competition | Postings vs applications per vacancy (four quadrants); job-seeker interest (views) vs supply |
| Pay vs friction | Reward (pay) vs barrier (experience) in the most competitive categories; pay premium vs repost rate in the biggest categories |
| Skills | Pay premium for 18 keywords in job titles |
| Career recommender | Category × seniority tracks ranked by user-weighted demand, pay, low competition and ease of entry |
| Sector table | Every category benchmarked against its broad sector, downloadable as CSV |
| Data quality | Checks on the filtered rows, and what each cleaning rule changed in the whole file |

## Project files

- `app.py` — the Streamlit dashboard.
- `data_utils.py` — loading, cleaning, features and summary tables. The app and the notebook both import it, so they cannot drift apart.
- `notebooks/01_eda.ipynb` — the EDA behind the dashboard (Parts 0–7). Run All after placing the CSV in `data/`; charts are saved to `figures/`.
- `report.md` — written submission.
- `presentation_script.md`, `presentation/` — 10-minute team presentation.

## Data handling rules

All rules live in `data_utils.py` as named constants.

- **Empty rows:** rows with no job id (3,988 in the full file) are dropped.
- **Dates:** ISO dates (`2023-04-08`) are read as year-month-day; other formats are read day-first.
- **Salary:**
  - zero means "not disclosed" and becomes missing;
  - reversed min/max ranges are swapped;
  - monthly values below S$500 or above S$60,000 are blanked, but the posting is kept;
  - the top and bottom 1% are capped, not deleted.
- **Categories:** a posting can belong to up to five categories. Demand, competition, the recommender and the sector table count it in each one; Overview, Salary and Trends use its first-listed category. The 43 categories are grouped into 7 broad sectors (`BROAD_SECTORS`).
- **PMET:** only `Non-executive` postings are Non-PMET (a simplification; the source has no PMET field).
- **Reposts:** `metadata_repostCount > 0` marks an advert put up more than once, a hint that it was hard to fill.
- **Engagement window:**
  - Views and applications stop building up for adverts first posted after June 2023.
  - Applications per vacancy and views per posting therefore use only adverts first posted up to `ENGAGEMENT_END` (30 June 2023).
- **Trends:** posting volume is complete only from May 2023 (`TREND_CUTOFF`), so trend views start there by default.

## Acknowledgement

The broad-sector, PMET and repost lenses, the reward-vs-barrier and friction-vs-premium views, the time-granularity selector and the benchmark table were adapted from the public [SG Job Market Trend Analysis dashboard](https://huggingface.co/spaces/Miqqie/job) on Hugging Face Spaces. They were rebuilt in pandas on our own cleaning rules. That dashboard divides applications and views by every posting; we keep our engagement-window rule instead.

## Course concepts used

- **SQL/database lessons:** structured fields and data-dictionary checks.
- **NumPy and pandas lessons:**
  - type conversion, `np.select` and `np.where`;
  - `explode`, `groupby().agg()`, `pivot_table`;
  - percentile ranks and date periods.
- **EDA lessons:** missing data, impossible values, duplicates, and collection artefacts that change what a metric means over time.
- **Visualisation lesson:**
  - chart chosen by the message;
  - labelled log axes;
  - one highlight colour;
  - captions that say what a chart cannot show.

## Before presenting

1. Re-run the notebook on the full file and update the numbers in `report.md`, `presentation_script.md` and the slides. Applications per vacancy now uses the engagement window, so the competition figures change.
2. Review missingness and salary quality before interpreting salary comparisons.
3. Capture 2–4 screenshots from the dashboard for the report, or show the app live.
4. Rehearse the handoffs and keep the presentation to 10 minutes.
