import time
import duckdb
import duckdb.typing as dt


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
    print("connecting to database...")
    conn = duckdb.connect()

    print("loading data...")
    rows = conn.read_json(
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

    print("converting to parquet...")
    rows.write_parquet(
        "lichess_db_eval.parquet",
        field_ids={
            "fen": 0,
            "cp": 1,
            "mate": 2,
        },
        compression="zstd",
    )

    print("done")


if __name__ == "__main__":
    main()
