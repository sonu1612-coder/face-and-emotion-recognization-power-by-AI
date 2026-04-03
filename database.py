import sqlite3
import os
from datetime import datetime

DB_NAME = "faces_dataset.db"

def init_db():
    """Initializes the SQLite database and creates the table if it does not exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            mask_status TEXT NOT NULL,
            emotion TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            identity TEXT DEFAULT 'Unknown'
        )
    ''')
    
    # Check if 'identity' column exists for older DB version migrations
    try:
        cursor.execute("SELECT identity FROM images LIMIT 1")
    except sqlite3.OperationalError:
        # identity column doesn't exist, alter table
        cursor.execute("ALTER TABLE images ADD COLUMN identity TEXT DEFAULT 'Unknown'")
        
    conn.commit()
    conn.close()

def insert_record(image_path, mask_status, emotion, identity="Unknown"):
    """Inserts a new face record into the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO images (image_path, mask_status, emotion, timestamp, identity)
        VALUES (?, ?, ?, ?, ?)
    ''', (image_path, mask_status, emotion, timestamp, identity))
    conn.commit()
    conn.close()

def get_all_records():
    """Retrieves all records from the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_path, mask_status, emotion, identity, timestamp FROM images ORDER BY timestamp DESC")
    records = cursor.fetchall()
    conn.close()
    return records

def delete_record(record_id):
    """Deletes a record by ID and optionally its physical file."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Fetch image path to delete file
    cursor.execute("SELECT image_path FROM images WHERE id=?", (record_id,))
    row = cursor.fetchone()
    if row and os.path.exists(row[0]):
        try:
            os.remove(row[0])
        except Exception as e:
            print(f"File delete err: {e}")
            
    cursor.execute("DELETE FROM images WHERE id=?", (record_id,))
    conn.commit()
    conn.close()
    return True

def update_record(record_id, identity):
    """Updates the identity of a stored face record."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE images SET identity=? WHERE id=?", (identity, record_id))
    conn.commit()
    conn.close()
    return True

def wipe_database():
    """Deletes all records from the database and clears the dataset folder."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM images")
    rows = cursor.fetchall()
    
    # Delete physiological files
    for r in rows:
        if os.path.exists(r[0]):
            try: os.remove(r[0])
            except Exception: pass
            
    # Purge Table
    cursor.execute("DELETE FROM images")
    conn.commit()
    conn.close()
    return True

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")

