import os
import sys
import sqlite3
import json
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_db_connection():
    """Establishes connection to SQLite database."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes SQLite database schema for prediction history."""
    os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predicted_code TEXT NOT NULL,
            predicted_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            is_low_confidence INTEGER NOT NULL,
            ranked_json TEXT NOT NULL,
            guidance TEXT NOT NULL,
            dermatologist_recommendation TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def save_to_supabase(record_data):
    """
    Sends prediction record payload to Supabase via PostgREST REST API.
    
    Args:
        record_data (dict): Standard prediction history record dictionary.
        
    Returns:
        dict: { "success": bool, "data": dict or None, "error": str or None }
    """
    if not (config.SUPABASE_URL and config.SUPABASE_KEY):
        return {"success": False, "error": "Supabase credentials not configured in environment."}

    endpoint = f"{config.SUPABASE_URL.rstrip('/')}/rest/v1/prediction_history"
    headers = {
        "apikey": config.SUPABASE_KEY,
        "Authorization": f"Bearer {config.SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

    payload = {
        "image_filename": record_data["image_filename"],
        "image_path": record_data["image_path"],
        "predicted_code": record_data["predicted_code"],
        "predicted_name": record_data["predicted_name"],
        "confidence": record_data["confidence"],
        "risk_level": record_data["risk_level"],
        "is_low_confidence": record_data["is_low_confidence"],
        "ranked_json": record_data["ranked_conditions"],
        "guidance": record_data["guidance"],
        "dermatologist_recommendation": record_data["dermatologist_recommendation"]
    }

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=5)
        if response.status_code in [200, 201]:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"Supabase API returned HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": f"Supabase connection error: {str(e)}"}

def save_prediction(result_data, image_url=""):
    """
    Saves a prediction result entry into the local SQLite database and optionally syncs with Supabase.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    top = result_data["top_prediction"]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO prediction_history (
            timestamp, image_filename, image_path, predicted_code, predicted_name,
            confidence, risk_level, is_low_confidence, ranked_json, guidance, dermatologist_recommendation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        now_str,
        result_data.get("image_name", "lesion.jpg"),
        image_url or result_data.get("image_name", "lesion.jpg"),
        top["code"],
        top["name"],
        top["confidence"],
        top["risk_level"],
        1 if result_data.get("is_low_confidence", False) else 0,
        json.dumps(result_data["ranked_conditions"]),
        top["guidance"],
        top["dermatologist_recommendation"]
    ))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Sync to Supabase if credentials are set
    if config.USE_SUPABASE:
        rec_data = {
            "image_filename": result_data.get("image_name", "lesion.jpg"),
            "image_path": image_url or result_data.get("image_name", "lesion.jpg"),
            "predicted_code": top["code"],
            "predicted_name": top["name"],
            "confidence": top["confidence"],
            "risk_level": top["risk_level"],
            "is_low_confidence": bool(result_data.get("is_low_confidence", False)),
            "ranked_conditions": result_data["ranked_conditions"],
            "guidance": top["guidance"],
            "dermatologist_recommendation": top["dermatologist_recommendation"]
        }
        sb_res = save_to_supabase(rec_data)
        if not sb_res["success"]:
            print(f"Supabase Sync Warning: {sb_res['error']}")

    return record_id

def get_prediction_history(limit=50):
    """Retrieves recent prediction history records (tries Supabase first if configured, falls back to SQLite)."""
    if config.USE_SUPABASE:
        try:
            endpoint = f"{config.SUPABASE_URL.rstrip('/')}/rest/v1/prediction_history?select=*&order=id.desc&limit={limit}"
            headers = {
                "apikey": config.SUPABASE_KEY,
                "Authorization": f"Bearer {config.SUPABASE_KEY}"
            }
            res = requests.get(endpoint, headers=headers, timeout=5)
            if res.status_code == 200:
                rows = res.json()
                records = []
                for row in rows:
                    records.append({
                        "id": row.get("id"),
                        "timestamp": row.get("timestamp", row.get("created_at", "")),
                        "image_filename": row.get("image_filename", ""),
                        "image_path": row.get("image_path", ""),
                        "predicted_code": row.get("predicted_code", ""),
                        "predicted_name": row.get("predicted_name", ""),
                        "confidence": row.get("confidence", 0),
                        "risk_level": row.get("risk_level", ""),
                        "is_low_confidence": bool(row.get("is_low_confidence", False)),
                        "ranked_conditions": row.get("ranked_json") if isinstance(row.get("ranked_json"), list) else json.loads(row.get("ranked_json", "[]")),
                        "guidance": row.get("guidance", ""),
                        "dermatologist_recommendation": row.get("dermatologist_recommendation", ""),
                        "source": "Supabase Cloud"
                    })
                return records
        except Exception as e:
            print(f"Supabase Fetch Warning: {e}, falling back to SQLite.")

    # SQLite fallback
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, timestamp, image_filename, image_path, predicted_code, predicted_name,
               confidence, risk_level, is_low_confidence, ranked_json, guidance, dermatologist_recommendation
        FROM prediction_history
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    records = []
    for row in rows:
        records.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "image_filename": row["image_filename"],
            "image_path": row["image_path"],
            "predicted_code": row["predicted_code"],
            "predicted_name": row["predicted_name"],
            "confidence": row["confidence"],
            "risk_level": row["risk_level"],
            "is_low_confidence": bool(row["is_low_confidence"]),
            "ranked_conditions": json.loads(row["ranked_json"]),
            "guidance": row["guidance"],
            "dermatologist_recommendation": row["dermatologist_recommendation"],
            "source": "Local SQLite"
        })
    return records

def delete_prediction_history(record_id):
    """Deletes a specific prediction history record by ID."""
    if config.USE_SUPABASE and record_id != "all":
        try:
            endpoint = f"{config.SUPABASE_URL.rstrip('/')}/rest/v1/prediction_history?id=eq.{record_id}"
            headers = {
                "apikey": config.SUPABASE_KEY,
                "Authorization": f"Bearer {config.SUPABASE_KEY}"
            }
            requests.delete(endpoint, headers=headers, timeout=5)
        except Exception as e:
            print(f"Supabase Delete Warning: {e}")

    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

def clear_all_history():
    """Clears all prediction history records."""
    if config.USE_SUPABASE:
        try:
            endpoint = f"{config.SUPABASE_URL.rstrip('/')}/rest/v1/prediction_history?id=gt.0"
            headers = {
                "apikey": config.SUPABASE_KEY,
                "Authorization": f"Bearer {config.SUPABASE_KEY}"
            }
            requests.delete(endpoint, headers=headers, timeout=5)
        except Exception as e:
            print(f"Supabase Clear Warning: {e}")

    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history")
    conn.commit()
    conn.close()
