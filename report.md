# Singapore Jobs Explorer — Project Report

## 1. Business case

- **Scenario:** Career coaches and job seekers in Singapore want a quick, trustworthy view of which roles are being advertised and what pay the adverts list.
- **Objective:** Help a coach or job seeker compare advertised role volume, listed pay, and competition across industries, career levels, and time, so that a first career conversation starts from evidence.
- **Value:** A filterable dashboard that answers "where are the jobs, what do they pay, and how competitive are they?" without reading a million adverts. It is an exploration tool. It does not predict hiring and does not show who was hired.

## 2. Data handling and process

- **Tools:** Python, Pandas, NumPy, Plotly, and Streamlit.
- **Data:** `data/SGJobData.csv` contains 1,048,585 rows and 22 columns, one row per job advert. The dashboard loads and cleans the full file once, then caches it.
- **Sampling first:** We checked shape, column types, and the range of each numeric column on a 50,000-row sample before running the full file.
- **Quality checks:** Missing values, repeated job IDs, impossible salaries, and date parsing. 3,988 rows (0.38%) are entirely empty apart from a zero salary; they are dropped. The remaining 1,044,597 job IDs are all unique, so no duplicate removal was needed. We keep the de-duplication step as a safeguard.
- **Cleaning:**
  - Zero salaries are treated as "not disclosed", not as pay of zero, so they are set to missing.
  - Where the minimum exceeds the maximum, the two values are swapped.
  - Salaries that are impossible for a monthly figure are set to missing rather than corrected.
  - Job titles are normalised (marketing prefixes such as "Urgent hiring!!!" and trailing pay phrases are removed) so that one role is counted once.
  - The `categories` field is a JSON array. A job can sit in several categories, so it is expanded: one row per posting per category.
- **Feature engineering:** Nine official position levels are collapsed into five seniority bands (Entry, Junior, Mid, Senior, Management). Salary is the provided average where present, otherwise the midpoint of the minimum and maximum. Posting trends use the original posting date, and only from May 2023 onward (see Limitations).
- **Result:** 1,044,597 cleaned postings from 53,150 organisations, offering 2,725,717 vacancies across 43 categories.

## 3. Key findings (full cleaned file)

- **Pay:** The median advertised monthly salary is **S$3,850**; the middle 50% of postings span **S$2,950–S$5,500**. The mean is about S$824 higher than the median, because a small number of very high adverts raise it. The median is therefore the headline figure.
- **Pay rises with seniority (median monthly):** Entry S$2,675 (250,269 postings); Junior S$3,150 (167,656); Mid S$4,000 (365,909); Senior S$5,000 (100,459); Management S$6,500 (160,304).
- **Size and pay differ:** Information Technology has the most postings in the cleaned data (140,866; median S$6,500). Admin / Secretarial is large (117,854) but has one of the lowest medians (S$2,900).
- **Competition varies widely:** Applications per vacancy (total applications ÷ total vacancies) range from about 0.15 in Personal Care / Beauty and 0.21 in F&B to about 1.5 in Marketing / Public Relations and 2.2 in Risk Management.
- **Entry-friendly roles:** About 36% of postings ask for one year or less of experience. Customer Service and Admin / Secretarial are above 50%.

## 4. Dashboard / app

- **Solution:** Streamlit web app.
  - **Filters:** broad sector, industry / category, career level, employment type, PMET status, an option to exclude reposted adverts, and a posting date range.
  - All charts and cards use the same filtered rows.
- **Summary cards:** job postings, vacancies, median listed salary, salary coverage, share of reposted adverts, and applications per vacancy.
- **Overview tab:** most-posted job titles (cleaned names), posting volume by category, and vacancies by broad sector.
- **Salary tab:** pay ladder by seniority (median with the middle half), and median listed salary for the top categories.
- **Trends tab:** vacancies per week, month, quarter or year, split by PMET status. Competition is shown only where it can be measured. Read with the May 2023 limitation in mind.
- **Demand vs competition tab:**
  - postings against applications per vacancy, in four quadrants;
  - job-seeker interest (views per posting) against vacancies.
- **Pay vs friction tab:**
  - reward (median pay) against barrier (years of experience asked) in the most competitive categories;
  - pay premium over the broad-sector median against the share of reposted adverts in the biggest categories.
- **Skills and Career recommender tabs:**
  - Skills shows the pay premium for keywords in job titles.
  - The recommender gives a user-weighted fit score for category × seniority tracks.
- **Sector table tab:** every category benchmarked against its broad sector, with a CSV download.
- **Data quality tab:** checks on the filtered rows and a log of what each cleaning rule changed. It lets a user check whether a chart is reliable before relying on it.
- **Acknowledgement:** the broad-sector, PMET, repost and benchmark views were adapted from the public [SG Job Market Trend Analysis dashboard](https://huggingface.co/spaces/Miqqie/job) and rebuilt in pandas on our cleaning rules.

## 5. Challenges, learnings, and next steps

- **Challenge — zero-coded salaries:** 3,988 rows were blank apart from a zero salary. Treating zero as pay would have dragged every pay summary down, so zero is treated as missing and the empty rows are removed.
- **Challenge — overlapping categories:** A job can belong to up to five categories.
  - The notebook, and the dashboard's demand, competition, recommender and sector-table views, count each posting once per category it belongs to. That is why IT shows 140,866.
  - The Overview, Salary and Trends tabs use only the first category listed, so their category counts are lower.
- **Challenge — data-collection changes:**
  - Posting volume is only complete from May 2023, so trend charts and growth figures start there.
  - View and application counts stop building up for adverts first posted after June 2023, so competition and interest metrics use only adverts first posted up to June 2023.
- **Challenge — date format:** the dates are ISO (`2023-04-08`). Reading them day-first turned 5 January into 1 May, scrambling the monthly trends. Dates are now read as year-month-day.
- **Learning:** A numeric zero can be a missing-data code. Salary, title, and date completeness must be checked together. A trend can reflect how data was collected rather than what happened in the market.
- **Limitation — outliers:** Some salaries are far outside the normal range (a raw maximum of S$25,330,000, and 279 raw postings with a maximum above S$100,000). These were not used for the pay medians, but the upper tail should be reviewed before anyone draws conclusions from it.
- **Limitation — adverts, not hires:** Posting counts and listed pay describe what employers advertised. They do not show hiring outcomes, applicant quality, or how many people are looking for work.
- **Next steps:** Review the extreme salaries one by one; add a clearer view of overlapping categories; test the app on the full file; and ask career coaches and job seekers to try the dashboard and report what is confusing.

## How to run

Copy `SGJobData.csv` into `data/`, install dependencies with `pip install -r requirements.txt`, then run `streamlit run app.py`. The notebook (`notebooks/01_eda.ipynb`) uses the same `data_utils.py` functions and runs end to end from the same CSV.

Presentation materials are in [presentation/](presentation/README.md).
