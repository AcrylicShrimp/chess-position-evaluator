mod write_chesseval;

use crate::write_chesseval::centipawn_to_win_prob;
use duckdb::{Connection, params};

const DUCKDB_TEMP_PATH: &str = "lichess_db_eval.duckdb.tmp";
const CHESS_EVALUATION_DB_PATH: &str = "lichess_db_eval.jsonl";
const TRAIN_CHESSEVAL_PATH: &str = "train.chesseval";
const VALIDATION_CHESSEVAL_PATH: &str = "validation.chesseval";
const DATASET_RATIO: f64 = 1.0;
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

    let dataset_size = (row_count as f64 * DATASET_RATIO) as i64;
    println!("will use {dataset_size} rows for training and validation");

    check_minimum_entropy(&conn)?;

    // 2. compute train and validation set sizes
    let validation_set_size = (dataset_size as f64 * VALIDATION_SET_RATIO) as i64;
    let train_set_size = dataset_size - validation_set_size;

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
        CREATE TABLE base_evals AS (
            SELECT 
                fen, 
                CASE 
                    WHEN pvs.mate IS NOT NULL THEN 
                        CASE WHEN pvs.mate > 0 THEN 2000 ELSE -2000 END
                    ELSE pvs.cp 
                END as cp
            FROM (
                SELECT fen, list_extract(eval.pvs, 1) as pvs
                FROM (
                    SELECT fen, unnest(evals) as eval
                    FROM read_json_auto(?1)
                )
                WHERE 24 <= eval.depth 
                  AND array_length(eval.pvs) != 0
            )
            WHERE (pvs.cp IS NOT NULL OR pvs.mate IS NOT NULL)
        )
        ",
    )?
    .execute(params![chess_evaluation_db_path])?;

    let nuanced_count = conn.query_row::<i64, _, _>(
        "SELECT COUNT(*) FROM base_evals WHERE cp >= -150 AND cp <= 150",
        params![],
        |row| row.get(0),
    )?;

    println!("Nuanced rows available: {}", nuanced_count);

    let side_ratio = 0.15;
    let limit_count = (nuanced_count as f64 * side_ratio) as i64;

    println!("Using ALL Nuanced rows.");
    println!(
        "Sampling {} rows for White/Black wins each (ratio {:.2})",
        limit_count, side_ratio
    );

    conn.prepare(
        "
        CREATE TABLE rows AS (
            SELECT fen, cp FROM (
                (SELECT fen, cp FROM base_evals WHERE cp > 150 ORDER BY RANDOM() LIMIT ?1)
                UNION ALL
                (SELECT fen, cp FROM base_evals WHERE cp < -150 ORDER BY RANDOM() LIMIT ?1)
                UNION ALL
                (SELECT fen, cp FROM base_evals WHERE cp >= -150 AND cp <= 150)
            )
            ORDER BY RANDOM()
        )
        ",
    )?
    .execute(params![limit_count])?;

    conn.execute("DROP TABLE base_evals", params![])?;

    Ok(())
}

fn binary_entropy(p: f32) -> f32 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
}

fn check_minimum_entropy(conn: &Connection) -> Result<(), anyhow::Error> {
    let mut stmt = conn.prepare("SELECT cp FROM rows")?;
    let cp_iter = stmt.query_map(params![], |row| row.get::<_, i32>(0))?;

    let mut total_entropy = 0f64;
    let mut count = 0;
    let mut zero_cp_count = 0;

    for cp_result in cp_iter {
        let cp = cp_result?;

        if cp == 0 {
            zero_cp_count += 1;
        }

        let prob = centipawn_to_win_prob(cp);
        let entropy = binary_entropy(prob);

        total_entropy += entropy as f64;
        count += 1;
    }

    let avg_entropy = total_entropy / count as f64;
    let zero_ratio = zero_cp_count as f64 / count as f64 * 100.0;

    println!("--------------------------------------------------");
    println!("Total rows analyzed: {}", count);
    println!("'Draw' (cp=0) ratio: {:.2}%", zero_ratio);
    println!("--------------------------------------------------");
    println!("★ Theoretical Minimum BCE Loss: {:.6}", avg_entropy);
    println!("--------------------------------------------------");
    println!("(Even if your model is PERFECT, the loss cannot go below this value)");

    Ok(())
}
