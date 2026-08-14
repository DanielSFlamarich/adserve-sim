WITH PARAMS AS (
    SELECT to_date('%s', 'YYYY-MM-DD') start_date
)
SELECT
    DATE(st.date) AS date,
FROM some_schema.some_table AS st,
params
GROUP BY
    DATE(st.date)
ORDER BY
    DATE(st.date) DESC
