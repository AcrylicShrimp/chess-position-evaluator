import duckdb
import plotly.express as px


def main():
    conn = duckdb.connect()
    cp_values = conn.sql(
        """
        SELECT cp
        FROM parquet_scan('lichess_db_eval.parquet')
        WHERE cp IS NOT NULL AND 10 <= ABS(cp)
        USING SAMPLE 100000 ROWS (reservoir)
        """
    )

    fig = px.histogram(cp_values, x="cp", nbins=60)
    fig.show()


if __name__ == "__main__":
    main()
