import duckdb


def main():
    conn = duckdb.connect("preprocess/lichess_db_eval.duckdb.tmp", True)
    total_count = conn.execute("SELECT COUNT(*) FROM rows").fetchone()[0]

    clipped_count = conn.execute(
        "SELECT COUNT(*) FROM rows WHERE 2000 < ABS(cp)"
    ).fetchone()[0]
    mean = conn.execute("SELECT AVG(cp) FROM rows WHERE cp IS NOT NULL").fetchone()[0]
    std = conn.execute("SELECT STDDEV(cp) FROM rows WHERE cp IS NOT NULL").fetchone()[0]
    min = conn.execute("SELECT MIN(cp) FROM rows WHERE cp IS NOT NULL").fetchone()[0]
    max = conn.execute("SELECT MAX(cp) FROM rows WHERE cp IS NOT NULL").fetchone()[0]
    white_is_winning_count = conn.execute(
        "SELECT COUNT(*) FROM rows WHERE cp > 0"
    ).fetchone()[0]
    black_is_winning_count = conn.execute(
        "SELECT COUNT(*) FROM rows WHERE cp < 0"
    ).fetchone()[0]
    equal_position_count = conn.execute(
        "SELECT COUNT(*) FROM rows WHERE cp = 0"
    ).fetchone()[0]

    print("Total count", total_count)
    print("Clipped count", clipped_count)
    print(f"Clipped ratio {clipped_count / total_count * 100:.2f}%")
    print("Cp Mean", mean)
    print("Cp Std", std)
    print("Cp Min", min)
    print("Cp Max", max)
    print("White is winning count", white_is_winning_count)
    print("Black is winning count", black_is_winning_count)
    print("Equal position count", equal_position_count)

    first_20_rows = conn.execute("SELECT cp FROM rows LIMIT 20").fetchall()
    mean = sum(row[0] for row in first_20_rows) / len(first_20_rows)
    std = (
        sum((row[0] - mean) ** 2 for row in first_20_rows) / len(first_20_rows)
    ) ** 0.5

    print("Mean of first 20 rows", mean)
    print("Std of first 20 rows", std)


if __name__ == "__main__":
    main()
