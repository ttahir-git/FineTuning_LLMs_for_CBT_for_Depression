import pandas as pd
from pathlib import Path

def find_patient_for_session(session_id, base_dir):
    """
    Search through the directory structure to find which patient's folder contains
    the session with the given ID.
    """
    # Convert to Path object for easier handling
    base_path = Path(base_dir)
    
    # Walk through all subdirectories
    for group_dir in base_path.iterdir():
        if not group_dir.is_dir():
            continue
            
        # Each group directory contains patient directories
        for patient_dir in group_dir.iterdir():
            if not patient_dir.is_dir():
                continue
                
            # Check if the directory name matches the expected pattern
            if "CBT_Depression_Simulations_" in patient_dir.name:
                # Look for the session file in this patient's directory
                for transcript in patient_dir.glob("**/*"):
                    if transcript.is_file() and session_id in transcript.name:
                        # Extract patient name from directory name
                        patient_name = patient_dir.name.replace("CBT_Depression_Simulations_", "")
                        return patient_name
    
    return None

def main():
    # Define paths
    base_dir = '/Users/talhatahir/Documents/GitHub/CBT_Project_Final/Cleaned_Simulation_Data'
    scores_path = '/Users/talhatahir/Documents/GitHub/CBT_Project_Final/Code/evaluation_results/all_scores.csv'
    output_path = '/Users/talhatahir/Documents/GitHub/CBT_Project_Final/Code/evaluation_results/all_scores_with_patients.csv'
    
    # Read the scores CSV
    df = pd.read_csv(scores_path)
    
    # Create a new patient_id column
    patient_ids = []
    
    print("Processing sessions...")
    
    # For each session ID, find the corresponding patient
    for session_id in df['id']:
        patient = find_patient_for_session(session_id, base_dir)
        patient_ids.append(patient)
        
    # Insert the patient_id column after the id column
    df.insert(1, 'patient_id', patient_ids)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_path, index=False)
    print(f"Updated CSV saved to: {output_path}")

if __name__ == "__main__":
    main()