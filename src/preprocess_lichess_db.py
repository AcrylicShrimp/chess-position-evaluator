import duckdb
import duckdb.typing as dt
import os


schema: dict[str, str] = {
    "fen": str(dt.DuckDBPyType(str)),
    "evals": str(
        dt.DuckDBPyType(
            list[
                dt.DuckDBPyType(
                    {
                        "depth": int,
                        "pvs": list[
                            dt.DuckDBPyType(
                                {
                                    "cp": int,
                                    "mate": int,
                                    "line": str,
                                }
                            )
                        ],
                    }
                )
            ]
        )
    ),
}


def main():
    if os.path.exists("lichess_db_eval.duckdb"):
        print("lichess_db_eval.duckdb already exists; removing...")
        os.remove("lichess_db_eval.duckdb")

    with duckdb.connect() as temp_conn:
        print("connected to database")

        rows = temp_conn.read_json(
            "lichess_db_eval.jsonl",
            columns=schema,
        )
        rows = rows.select("fen, unnest(evals) as eval")
        rows = rows.filter("10 <= eval.depth AND array_length(eval.pvs) != 0")
        rows = rows.select("fen, list_extract(eval.pvs, 1) as pvs")
        rows = rows.select("fen, pvs.cp as cp, pvs.mate as mate")
        rows = rows.filter("(cp IS NOT NULL AND 100 <= ABS(cp)) OR (mate IS NOT NULL)")

        print("loaded:")
        print(rows)

        total_rows = rows.count("*").fetchone()[0]
        print(f"total rows: {total_rows}")

        validation_set_ratio = 0.1
        validation_set_size = int(total_rows * validation_set_ratio)
        train_set_size = total_rows - validation_set_size

        print(f"validation set size: {validation_set_size}")
        print(f"train set size: {train_set_size}")

        print("creating train and validation sets...")
        rows = rows.order("RANDOM()")
        rows = rows.to_arrow_table()

    with duckdb.connect("lichess_db_eval.duckdb") as conn:
        conn.register("shuffled_view", rows)
        conn.sql(
            """
            CREATE TABLE train_rows AS
            SELECT ROW_NUMBER() OVER () AS row_idx, fen, cp, mate
            FROM shuffled_view
            LIMIT $train_set_size
            """,
            params={
                "train_set_size": train_set_size,
            },
        )
        conn.sql(
            """
            CREATE TABLE validation_rows AS
            SELECT ROW_NUMBER() OVER () AS row_idx, fen, cp, mate
            FROM shuffled_view
            OFFSET $train_set_size
            LIMIT $validation_set_size
            """,
            params={
                "train_set_size": train_set_size,
                "validation_set_size": validation_set_size,
            },
        )

        conn.sql(
            """
            CREATE INDEX train_rows_idx ON train_rows (row_idx)
            """
        )
        conn.sql(
            """
            CREATE INDEX validation_rows_idx ON validation_rows (row_idx)
            """
        )

    print("done")


if __name__ == "__main__":
    main()
