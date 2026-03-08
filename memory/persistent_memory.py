import sqlite3
import json
import os
from config.settings import MEMORY_DB_PATH

class PersistentMemory:
    def __init__(self):
        self.db_path = MEMORY_DB_PATH
        self._initialize_db()

    def _initialize_db(self):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                original_input TEXT,
                parsed_problem TEXT,
                retrieved_context TEXT,
                generated_solution TEXT,
                verification_result TEXT,
                user_feedback TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

    def save_interaction(self, session_id: str, data: dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO interactions (
                session_id, original_input, parsed_problem,
                retrieved_context, generated_solution, verification_result, user_feedback
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            data.get("original_input", ""),
            json.dumps(data.get("parsed_problem", {})),
            json.dumps(data.get("retrieved_context", [])),
            data.get("generated_solution", ""),
            json.dumps(data.get("verification_result", {})),
            data.get("user_feedback", "")
        ))
        
        conn.commit()
        conn.close()
        
    def update_feedback(self, session_id: str, feedback: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE interactions 
            SET user_feedback = ? 
            WHERE session_id = ?
        ''', (feedback, session_id))
        
        conn.commit()
        conn.close()

    def get_history(self, limit: int = 10):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM interactions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
