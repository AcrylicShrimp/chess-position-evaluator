import duckdb


def main():
    conn = duckdb.connect("preprocess/lichess_db_eval.duckdb.tmp")
    total_count = conn.execute("SELECT COUNT(*) FROM rows").fetchone()[0]

    clipped_count = conn.execute(
        "SELECT COUNT(*) FROM rows WHERE 2000 < ABS(cp)"
    ).fetchone()[0]
    mean = conn.execute("SELECT AVG(cp) FROM rows WHERE cp IS NOT NULL").fetchone()[0]
    std = conn.execute("SELECT STDDEV(cp) FROM rows WHERE cp IS NOT NULL").fetchone()[0]
    min = conn.execute("SELECT MIN(cp) FROM rows WHERE cp IS NOT NULL").fetchone()[0]
    max = conn.execute("SELECT MAX(cp) FROM rows WHERE cp IS NOT NULL").fetchone()[0]

    print("Total count", total_count)
    print("Clipped count", clipped_count)
    print(f"Clipped ratio {clipped_count / total_count * 100:.2f}%")
    print("Cp Mean", mean)
    print("Cp Std", std)
    print("Cp Min", min)
    print("Cp Max", max)


if __name__ == "__main__":
    main()
