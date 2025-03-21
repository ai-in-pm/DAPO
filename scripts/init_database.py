import os
import argparse
import sys
import logging

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import DAPODatabase
from utils.logging import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Initialize the DAPO database")
    parser.add_argument(
        '--db-path', type=str, default='data/dapo.db',
        help='Path to the database file'
    )
    parser.add_argument(
        '--import', dest='import_file', type=str, default=None,
        help='Import samples from a JSONL file'
    )
    parser.add_argument(
        '--split', type=str, default='train',
        help='Dataset split for imported samples (train, eval, test)'
    )
    args = parser.parse_args()
    
    # Setup logger
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger('init_database', log_dir)
    
    # Initialize database
    logger.info(f"Initializing database at {args.db_path}")
    database = DAPODatabase(args.db_path)
    
    # Import samples if specified
    if args.import_file:
        if not os.path.exists(args.import_file):
            logger.error(f"Import file not found: {args.import_file}")
            return
            
        logger.info(f"Importing samples from {args.import_file} as {args.split}")
        count = database.import_training_samples(args.import_file, args.split)
        logger.info(f"Imported {count} samples")
    
    logger.info("Database initialization completed")

if __name__ == '__main__':
    main()
