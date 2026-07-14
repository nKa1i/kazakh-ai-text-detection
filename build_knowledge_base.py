import sqlite3
import os

def init_db(db_path="data/knowledge_base.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS merchants")
    cur.execute("CREATE TABLE merchants (name TEXT PRIMARY KEY, category TEXT, amenities TEXT)")
    
    # Seed Ground Truth Records
    data = [
        ("Kaspi Coffee Almaty", "Cafe", "wifi, takeaway, terrace"),
        ("Kaspi Electronics", "Store", "delivery, warranty, credit"),
        ("Kaspi Hotel Astana", "Hotel", "wifi, breakfast, parking")
    ]
    cur.executemany("INSERT INTO merchants VALUES (?, ?, ?)", data)
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

if __name__ == "__main__":
    init_db()
