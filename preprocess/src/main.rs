mod write_chesseval;

use duckdb::{Connection, params};

const DUCKDB_TEMP_PATH: &str = "lichess_db_eval.duckdb.tmp";
const CHESS_EVALUATION_DB_PATH: &str = "lichess_db_eval.jsonl";
const TRAIN_CHESSEVAL_PATH: &str = "train.chesseval";
const VALIDATION_CHESSEVAL_PATH: &str = "validation.chesseval";
const VALIDATION_SET_RATIO: f64 = 0.1;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    if tokio::fs::try_exists(TRAIN_CHESSEVAL_PATH).await? {
        println!("{TRAIN_CHESSEVAL_PATH} already exists; removing...");
        tokio::fs::remove_file(TRAIN_CHESSEVAL_PATH).await?;
    }

    if tokio::fs::try_exists(VALIDATION_CHESSEVAL_PATH).await? {
        println!("{VALIDATION_CHESSEVAL_PATH} already exists; removing...");
        tokio::fs::remove_file(VALIDATION_CHESSEVAL_PATH).await?;
    }

    create_temp_table(DUCKDB_TEMP_PATH, CHESS_EVALUATION_DB_PATH).await?;

    let conn = Connection::open(DUCKDB_TEMP_PATH)?;
    let row_count = conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM rows", params![], |row| {
        row.get::<_, i64>(0)
    })?;
    println!("total {row_count} rows loaded");

    // 2. compute train and validation set sizes
    let validation_set_size = (row_count as f64 * VALIDATION_SET_RATIO) as i64;
    let train_set_size = row_count - validation_set_size;

    println!("train set size: {train_set_size}");
    println!("validation set size: {validation_set_size}");

    drop(conn);

    println!("writing train set to {TRAIN_CHESSEVAL_PATH}");
    write_chesseval::write_chesseval(DUCKDB_TEMP_PATH, TRAIN_CHESSEVAL_PATH, 0, train_set_size)
        .await?;

    println!("writing validation set to {VALIDATION_CHESSEVAL_PATH}");
    write_chesseval::write_chesseval(
        DUCKDB_TEMP_PATH,
        VALIDATION_CHESSEVAL_PATH,
        train_set_size,
        validation_set_size,
    )
    .await?;

    println!("done");

    Ok(())
}

async fn create_temp_table(
    path: &str,
    chess_evaluation_db_path: &str,
) -> Result<(), anyhow::Error> {
    if tokio::fs::try_exists(path).await? {
        println!("{path} already exists; reusing...");
        return Ok(());
    }

    let conn = Connection::open(path)?;

    conn.prepare(
        "
        CREATE TABLE all_rows AS (
            SELECT fen, pvs.cp as cp
            FROM (
                SELECT fen, list_extract(eval.pvs, 1) as pvs
                FROM (
                    SELECT fen, unnest(evals) as eval
                    FROM read_json_auto(?1)
                )
                WHERE 10 <= eval.depth AND array_length(eval.pvs) != 0
            )
            WHERE pvs.cp IS NOT NULL AND 50 <= ABS(pvs.cp) AND ABS(pvs.cp) <= 2000
            ORDER BY RANDOM()
        )
        ",
    )?
    .execute(params![chess_evaluation_db_path])?;

    let white_is_winning_count = conn.query_row::<i64, _, _>(
        "SELECT COUNT(*) FROM all_rows WHERE cp > 0",
        params![],
        |row| row.get::<_, i64>(0),
    )?;
    let black_is_winning_count = conn.query_row::<i64, _, _>(
        "SELECT COUNT(*) FROM all_rows WHERE cp < 0",
        params![],
        |row| row.get::<_, i64>(0),
    )?;
    let minimum = white_is_winning_count.min(black_is_winning_count);
    let base_total_count = white_is_winning_count + black_is_winning_count;

    let equal_position_count = (base_total_count as f32 * 0.2) as i64;
    let white_wins_count = (base_total_count as f32 * 0.1) as i64;
    let black_wins_count = (base_total_count as f32 * 0.1) as i64;

    conn.prepare(
        "
        INSERT INTO all_rows
        SELECT fen, pvs.cp as cp
        FROM (
            SELECT fen, list_extract(eval.pvs, 1) as pvs
            FROM (
                SELECT fen, unnest(evals) as eval
                FROM read_json_auto(?1)
            )
            WHERE 10 <= eval.depth AND array_length(eval.pvs) != 0
        )
        WHERE pvs.cp IS NOT NULL AND ABS(pvs.cp) <= 10
        ORDER BY RANDOM()
        LIMIT ?2
        ",
    )?
    .execute(params![chess_evaluation_db_path, equal_position_count])?;
    conn.prepare(
        "
        INSERT INTO all_rows
        SELECT fen, 2000 as cp
        FROM (
            SELECT fen, list_extract(eval.pvs, 1) as pvs
            FROM (
                SELECT fen, unnest(evals) as eval
                FROM read_json_auto(?1)
            )
            WHERE 10 <= eval.depth AND array_length(eval.pvs) != 0
        )
        WHERE pvs.mate IS NOT NULL AND 1 <= pvs.mate AND pvs.mate <= 3
        ORDER BY RANDOM()
        LIMIT ?2
        ",
    )?
    .execute(params![chess_evaluation_db_path, white_wins_count])?;
    conn.prepare(
        "
        INSERT INTO all_rows
        SELECT fen, -2000 as cp
        FROM (
            SELECT fen, list_extract(eval.pvs, 1) as pvs
            FROM (
                SELECT fen, unnest(evals) as eval
                FROM read_json_auto(?1)
            )
            WHERE 10 <= eval.depth AND array_length(eval.pvs) != 0
        )
        WHERE pvs.mate IS NOT NULL AND -3 <= pvs.mate AND pvs.mate <= -1
        ORDER BY RANDOM()
        LIMIT ?2
        ",
    )?
    .execute(params![chess_evaluation_db_path, black_wins_count])?;

    conn.prepare(
        "
        CREATE TABLE white_wins AS (
            SELECT fen, cp
            FROM all_rows
            WHERE cp > 0
            ORDER BY RANDOM()
        )
        ",
    )?
    .execute(params![])?;
    conn.prepare(
        "
        CREATE TABLE black_wins AS (
            SELECT fen, cp
            FROM all_rows
            WHERE cp < 0
            ORDER BY RANDOM()
        )
        ",
    )?
    .execute(params![])?;
    conn.prepare(
        "
        CREATE TABLE equal_positions AS (
            SELECT fen, cp
            FROM all_rows
            WHERE cp = 0
            ORDER BY RANDOM()
        )
        ",
    )?
    .execute(params![])?;

    conn.prepare(
        "
        CREATE TABLE rows AS (
            SELECT fen, cp
            FROM (
                (SELECT fen, cp FROM white_wins LIMIT ?1)
                UNION ALL
                (SELECT fen, cp FROM black_wins LIMIT ?1)
                UNION ALL
                (SELECT fen, cp FROM equal_positions)
            )
            ORDER BY RANDOM()
        )
        ",
    )?
    .execute(params![minimum])?;

    Ok(())
}
