import sqlite3
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

class DAPODatabase:
    """Database for DAPO agent interactions and training data."""
    
    def __init__(self, db_path: str):
        """Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.logger = logging.getLogger('dapo_database')
        
        # Create database directory if it doesn't exist
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        # Initialize database
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            responses TEXT NOT NULL,
            answer_key TEXT,
            metadata TEXT,
            split TEXT DEFAULT 'train'
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            metadata TEXT
        )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection.
        
        Returns:
            SQLite connection object.
        """
        return sqlite3.connect(self.db_path)
    
    def save_interaction(self, prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Save an interaction to the database.
        
        Args:
            prompt: User prompt text.
            response: Agent response text.
            metadata: Optional metadata dictionary.
            
        Returns:
            ID of the inserted interaction.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute(
            "INSERT INTO interactions (prompt, response, timestamp, metadata) VALUES (?, ?, ?, ?)",
            (prompt, response, timestamp, metadata_json)
        )
        
        interaction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return interaction_id
    
    def get_interactions(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get recent interactions from the database.
        
        Args:
            limit: Maximum number of interactions to retrieve.
            offset: Offset for pagination.
            
        Returns:
            List of interaction dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, prompt, response, timestamp, metadata FROM interactions ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        
        interactions = []
        for row in cursor.fetchall():
            interaction = {
                'id': row[0],
                'prompt': row[1],
                'response': row[2],
                'timestamp': row[3],
                'metadata': json.loads(row[4]) if row[4] else None
            }
            interactions.append(interaction)
            
        conn.close()
        return interactions
    
    def save_training_sample(self, prompt: str, responses: List[str], 
                           answer_key: Optional[Any] = None, 
                           metadata: Optional[Dict[str, Any]] = None,
                           split: str = 'train') -> int:
        """Save a training sample to the database.
        
        Args:
            prompt: Prompt text.
            responses: List of response texts.
            answer_key: Optional answer key for the prompt.
            metadata: Optional metadata dictionary.
            split: Dataset split (train, eval, test).
            
        Returns:
            ID of the inserted sample.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        responses_json = json.dumps(responses)
        answer_key_json = json.dumps(answer_key) if answer_key is not None else None
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute(
            "INSERT INTO training_samples (prompt, responses, answer_key, metadata, split) "
            "VALUES (?, ?, ?, ?, ?)",
            (prompt, responses_json, answer_key_json, metadata_json, split)
        )
        
        sample_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return sample_id
    
    def get_training_samples(self, split: str = 'train', limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """Get training samples from the database.
        
        Args:
            split: Dataset split (train, eval, test).
            limit: Maximum number of samples to retrieve.
            offset: Offset for pagination.
            
        Returns:
            List of sample dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, prompt, responses, answer_key, metadata FROM training_samples "
            "WHERE split = ? LIMIT ? OFFSET ?",
            (split, limit, offset)
        )
        
        samples = []
        for row in cursor.fetchall():
            sample = {
                'id': row[0],
                'prompt': row[1],
                'responses': json.loads(row[2]),
                'answer_key': json.loads(row[3]) if row[3] else None,
                'metadata': json.loads(row[4]) if row[4] else None
            }
            samples.append(sample)
            
        conn.close()
        return samples
    
    def export_training_samples(self, output_path: str, split: str = 'train') -> int:
        """Export training samples to a JSONL file.
        
        Args:
            output_path: Path to the output JSONL file.
            split: Dataset split to export.
            
        Returns:
            Number of exported samples.
        """
        samples = self.get_training_samples(split=split, limit=100000)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                # Convert to the expected format for the dataset
                output_item = {
                    'prompt': sample['prompt'],
                    'responses': sample['responses'],
                    'answer_key': sample['answer_key']
                }
                
                # Add any metadata if it exists
                if sample['metadata']:
                    output_item.update(sample['metadata'])
                    
                f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                count += 1
        
        return count
    
    def import_training_samples(self, input_path: str, split: str = 'train') -> int:
        """Import training samples from a JSONL file.
        
        Args:
            input_path: Path to the input JSONL file.
            split: Dataset split to assign to the imported samples.
            
        Returns:
            Number of imported samples.
        """
        if not os.path.exists(input_path):
            self.logger.error(f"Input file not found: {input_path}")
            return 0
        
        count = 0
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Extract fields
                    prompt = data.get('prompt', '')
                    responses = data.get('responses', [])
                    answer_key = data.get('answer_key', None)
                    
                    # Extract metadata (all other fields)
                    metadata = {k: v for k, v in data.items() 
                               if k not in ['prompt', 'responses', 'answer_key']}
                    
                    # Save to database
                    self.save_training_sample(prompt, responses, answer_key, metadata, split)
                    count += 1
                    
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON line in {input_path}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error importing line: {e}")
                    continue
        
        return count
    
    def save_metric(self, metric_name: str, metric_value: float, 
                   metadata: Optional[Dict[str, Any]] = None) -> int:
        """Save a metric to the database.
        
        Args:
            metric_name: Name of the metric.
            metric_value: Value of the metric.
            metadata: Optional metadata dictionary.
            
        Returns:
            ID of the inserted metric.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute(
            "INSERT INTO metrics (timestamp, metric_name, metric_value, metadata) VALUES (?, ?, ?, ?)",
            (timestamp, metric_name, metric_value, metadata_json)
        )
        
        metric_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return metric_id
    
    def get_metrics(self, metric_name: Optional[str] = None, 
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get metrics from the database.
        
        Args:
            metric_name: Filter by metric name (None for all metrics).
            limit: Maximum number of metrics to retrieve.
            
        Returns:
            List of metric dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if metric_name is not None:
            cursor.execute(
                "SELECT id, timestamp, metric_name, metric_value, metadata FROM metrics "
                "WHERE metric_name = ? ORDER BY timestamp DESC LIMIT ?",
                (metric_name, limit)
            )
        else:
            cursor.execute(
                "SELECT id, timestamp, metric_name, metric_value, metadata FROM metrics "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        
        metrics = []
        for row in cursor.fetchall():
            metric = {
                'id': row[0],
                'timestamp': row[1],
                'metric_name': row[2],
                'metric_value': row[3],
                'metadata': json.loads(row[4]) if row[4] else None
            }
            metrics.append(metric)
            
        conn.close()
        return metrics
