import sqlite3
import struct

DB_PATH = 'predictions.db'


def safe_float_from_bytes(b):
    """
    Unpack IEEE 754 float32 from little-endian bytes blob
    """
    if isinstance(b, (bytes, bytearray, memoryview)) and len(b) == 4:
        try:
            return struct.unpack('<f', b)[0]
        except:
            return 0.0

    try:
        return float(b)
    except:
        return 0.0


def repair_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Count total rows
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE id IS NOT NULL")
    total = cursor.fetchone()[0]
    print(f"Total predictions: {total}")

    float_cols = [
        'prob_high',
        'prob_medium',
        'prob_low',
        'pass_probability',
        'score_estimate',
        'attendance'
    ]

    updated = 0

    # Bulk fix NULL or BLOB values
    for col in float_cols:
        cursor.execute(f"""
            UPDATE predictions
            SET {col} = ?
            WHERE typeof({col}) IN ('blob', 'null') OR {col} IS NULL
        """, (0.0,))
        
        updated += cursor.rowcount
        print(f"Fixed {cursor.rowcount} values in column: {col}")

    # Precise fix for BLOB values in prob_high
    cursor.execute("""
        SELECT id, prob_high 
        FROM predictions 
        WHERE typeof(prob_high) = 'blob'
    """)

    for row_id, bad_val in cursor.fetchall():
        fixed_val = safe_float_from_bytes(bad_val)
        cursor.execute("""
            UPDATE predictions 
            SET prob_high = ? 
            WHERE id = ?
        """, (fixed_val, row_id))
        updated += 1

    print(f"Precise prob_high repairs: {updated}")

    conn.commit()
    conn.close()

    print("✅ DB repair complete. Run `rm fix_db.py` after verification.")


if __name__ == '__main__':
    repair_db()