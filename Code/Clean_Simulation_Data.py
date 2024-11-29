import os
import logging
from datetime import datetime

def setup_logging(base_dir):
    """Set up logging configuration"""
    os.makedirs(base_dir, exist_ok=True)
    
    log_file = os.path.join(base_dir, 'cleaning_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directory_structure(source_dir, target_dir):
    """Recreate the directory structure in the target directory"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        logging.info(f"Created main directory: {target_dir}")

    for root, dirs, _ in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        if relative_path == '.':
            continue
            
        target_path = os.path.join(target_dir, relative_path)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            logging.info(f"Created directory: {target_path}")

def clean_transcript(content, file_path):
    """Clean the transcript content according to specifications"""
    lines = content.strip().split('\n')
    if not lines:
        logging.info(f"Empty file found: {file_path}")
        return None

    # Find indices of patient and therapist messages
    patient_indices = [i for i, line in enumerate(lines) 
                      if line.strip().startswith('Patient:')]
    therapist_indices = [i for i, line in enumerate(lines) 
                        if line.strip().startswith('Therapist:')]
    
    if not patient_indices or not therapist_indices:
        logging.info(f"Missing patient or therapist messages in: {file_path}")
        return None

    if len(patient_indices) < 1:
        logging.info(f"No patient messages found in: {file_path}")
        return None
    
    if len(therapist_indices) < 1:
        logging.info(f"No therapist messages found in: {file_path}")
        return None

    # Log the structure of the file
    logging.debug(f"File {file_path} has {len(lines)} lines, {len(patient_indices)} patient messages, and {len(therapist_indices)} therapist messages")
    
    # Remove first patient message and everything before it
    start_idx = patient_indices[0] + 1
    # Remove last therapist message and everything after it
    end_idx = therapist_indices[-1]
    
    cleaned_lines = lines[start_idx:end_idx]
    
    if not cleaned_lines:
        logging.info(f"No content remains after cleaning in: {file_path}")
        return None
        
    # Log the content that's being removed
    logging.debug(f"Removed first patient message: {lines[patient_indices[0]]}")
    logging.debug(f"Removed last therapist message: {lines[therapist_indices[-1]]}")
    
    return '\n'.join(cleaned_lines)

def process_files(source_dir, target_dir):
    """Process all transcript files in the directory structure"""
    files_processed = 0
    files_skipped = 0
    empty_files = 0

    for root, _, files in os.walk(source_dir):
        for file in files:
            if not file.endswith('.txt'):
                continue

            source_file = os.path.join(root, file)
            relative_path = os.path.relpath(root, source_dir)
            target_file = os.path.join(target_dir, relative_path, file)

            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Skip empty files
                if not content.strip():
                    logging.info(f"Skipping empty file: {source_file}")
                    empty_files += 1
                    continue

                cleaned_content = clean_transcript(content, source_file)
                
                # Skip if cleaning results in empty content
                if not cleaned_content:
                    logging.info(f"Skipping file after cleaning (no valid content): {source_file}")
                    files_skipped += 1
                    continue

                # Write cleaned content to new file
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                
                logging.info(f"Successfully processed: {source_file}")
                files_processed += 1

            except Exception as e:
                logging.error(f"Error processing {source_file}: {str(e)}")
                files_skipped += 1

    return files_processed, files_skipped, empty_files

def main():
    # Define directories
    source_dir = '/Users/talhatahir/Documents/GitHub/CBT_Project_Final/CBT_Simulations'
    base_dir = '/Users/talhatahir/Documents/GitHub/CBT_Project_Final'
    target_dir = os.path.join(base_dir, 'Cleaned_Simulation_Data')

    # Setup logging
    logger = setup_logging(target_dir)
    logger.info("Starting CBT transcript cleaning process")

    # Create directory structure
    create_directory_structure(source_dir, target_dir)

    # Process files
    start_time = datetime.now()
    files_processed, files_skipped, empty_files = process_files(source_dir, target_dir)
    end_time = datetime.now()

    # Log summary
    logger.info("\nProcessing Summary:")
    logger.info(f"Total files processed successfully: {files_processed}")
    logger.info(f"Total files skipped: {files_skipped}")
    logger.info(f"Empty files found: {empty_files}")
    logger.info(f"Total time taken: {end_time - start_time}")
    logger.info("Cleaning process completed")

if __name__ == "__main__":
    main()