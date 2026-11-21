import sqlite3
import pandas as pd

DB_PATH = "campaigns.db"


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_name TEXT NOT NULL,
                target_segment TEXT,
                contact_channel TEXT,
                start_date TEXT,
                budget REAL,
                target_customers INTEGER,
                objective TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def add_campaign(name, segment, channel, start_date, budget, target_customers, objective):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO campaigns
            (campaign_name, target_segment, contact_channel, start_date,
             budget, target_customers, objective)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, segment, channel, start_date, budget, target_customers, objective))
        conn.commit()


def get_campaigns():
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM campaigns ORDER BY id DESC", conn)


def delete_campaign(campaign_id):
    with get_connection() as conn:
        conn.execute("DELETE FROM campaigns WHERE id = ?", (campaign_id,))
        conn.commit()
