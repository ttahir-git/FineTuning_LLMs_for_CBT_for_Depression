import os
import json
import re

def clean_transcript(transcript):
    """Remove lines containing 'here is part x' from the transcript"""
    lines = transcript.split('\n')
    cleaned_lines = [line for line in lines if not re.search(r'here is part [1-4]', line, re.IGNORECASE)]
    return '\n'.join(cleaned_lines)

def process_transcripts(main_directory):
    """Process and combine transcripts from all relevant session files in the directory structure."""
    all_transcripts = []
    processed_files = []
    
    for dirpath, dirnames, filenames in os.walk(main_directory):
        if os.path.basename(dirpath).startswith('depression_set_'):
            session_files = [f for f in filenames 
                           if f.startswith('session_') 
                           and f.endswith('.json')
                           and '_part_' not in f]
            
            for session_file in session_files:
                file_path = os.path.join(dirpath, session_file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'transcript' in data:
                            transcript = data['transcript']
                            cleaned_transcript = clean_transcript(transcript)
                            all_transcripts.append(cleaned_transcript)
                            print(f"Processing: {os.path.join(os.path.basename(dirpath), session_file)}")
                            processed_files.append(os.path.join(os.path.basename(dirpath), session_file))
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    combined_text = '</conversation>\n'.join(all_transcripts)
    combined_text += '</conversation>'
    
    output_path = os.path.join(main_directory, 'Combined_Synthetic_Transcripts_Nov_2.txt')
    with open(output_path, 'w') as f:
        f.write(combined_text)
    
    print(f"\nSuccessfully combined {len(all_transcripts)} transcripts into {output_path}")
    print("\nProcessed files:")
    for file in sorted(processed_files):
        print(f"- {file}")

if __name__ == "__main__":
    main_directory = '/Users/talhatahir/Documents/GitHub/CBT_Project_Final/Synthetic_Transcripts'
    process_transcripts(main_directory)