import os
import asyncio
import logging
import json
from datetime import datetime
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re
import random
from collections import defaultdict
from asyncio import Semaphore
import time
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.generative_models._generative_models import SafetySetting

LOG_DIRECTORY = "logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)
LOG_FILENAME = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_PATH = os.path.join(LOG_DIRECTORY, LOG_FILENAME)
NUM_INSTANCES = 2
DATA_DIR = '/Users/talhatahir/Documents/GitHub/CBT_Project_Final/Cleaned_Simulation_Data'

EVALUATION_RESULTS_DIR = "evaluation_results"
os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)

REQUESTS_PER_MINUTE = 10  
MAX_CONCURRENT_REQUESTS = 1  
RATE_LIMIT_WINDOW = 60  

# Initialize Vertex AI
vertexai.init(project="***", location="northamerica-northeast1")
model = GenerativeModel("gemini-1.5-pro-002")

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.1,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    ),
]

class RateLimiter:
    """Rate limiter to control API request frequency"""
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.window = RATE_LIMIT_WINDOW
        self.tokens = requests_per_minute
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            time_passed = now - self.last_update
            self.last_update = now
            
            self.tokens = min(
                self.requests_per_minute,
                self.tokens + time_passed * (self.requests_per_minute / self.window)
            )
            
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) * (self.window / self.requests_per_minute)
                await asyncio.sleep(sleep_time)
                self.tokens = 1
            
            self.tokens -= 1

class NoRequestFilter(logging.Filter):
    """Filter to remove HTTP request logs"""
    def filter(self, record):
        return not record.getMessage().startswith(
            ("HTTP Request:", "connect_tcp", "start_tls", "send_request")
        )

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_PATH, mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.addFilter(NoRequestFilter())
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

WORD_TO_NUMBER = {
    "Severely Deficient": 1,
    "Very Poor": 2,
    "Poor": 3,
    "Unsatisfactory": 4,
    "Below Average": 5,
    "Adequate": 6,
    "Good": 7,
    "Very Good": 8,
    "Excellent": 9,
    "Outstanding": 10
}

NUMBER_TO_WORD = {v: k for k, v in WORD_TO_NUMBER.items()}
LOWER_WORD_TO_NUMBER = {k.lower(): v for k, v in WORD_TO_NUMBER.items()}
LOWER_NUMBER_TO_WORD = {k: v for k, v in NUMBER_TO_WORD.items()}

CATEGORIES = [
    "Agenda",
    "Feedback",
    "Understanding",
    "Interpersonal Effectiveness",
    "Collaboration",
    "Pacing and Efficient Use of Time",
    "Guided Discovery",
    "Focusing on Key Cognitions or Behaviors",
    "Strategy for Change",
    "Application of Cognitive-Behavioral Techniques",
    "Homework"
]

CTRS_CRITERIA = {
    "Agenda": """
Severely Deficient: Therapist did not set agenda.
Poor: Therapist set agenda that was vague or incomplete.
Good: Therapist worked with patient to set a mutually satisfactory agenda that included specific target problems (e.g., anxiety at work, dissatisfaction with marriage.)
Outstanding: Therapist worked with patient to set an appropriate agenda with target problems, suitable for the available time. Established priorities and then followed agenda.""",

    "Feedback": """
Severely Deficient: Therapist did not ask for feedback to determine patient's understanding of, or response to, the session.
Poor: Therapist elicited some feedback from the patient, but did not ask enough questions to be sure the patient understood the therapist's line of reasoning during the session or to ascertain whether the patient was satisfied with the session.
Good: Therapist asked enough questions to be sure that the patient understood the therapist's line of reasoning throughout the session and to determine the patient's reactions to the session. The therapist adjusted his/her behavior in response to the feedback, when appropriate.
Outstanding: Therapist was especially adept at eliciting and responding to verbal and non-verbal feedback throughout the session (e.g., elicited reactions to session, regularly checked for understanding, helped summarize main points at end of session).""",

    "Understanding": """
Severely Deficient: Therapist repeatedly failed to understand what the patient explicitly said and thus consistently missed the point. Poor empathic skills.
Poor: Therapist was usually able to reflect or rephrase what the patient explicitly said, but repeatedly failed to respond to more subtle communication. Limited ability to listen and empathize.
Good: Therapist generally seemed to grasp the patient's "internal reality" as reflected by both what the patient explicitly said and what the patient communicated in more subtle ways. Good ability to listen and empathize.
Outstanding: Therapist seemed to understand the patient's "internal reality" thoroughly and was adept at communicating this understanding through appropriate verbal and non-verbal responses to the patient (e.g., the tone of the therapist's response conveyed a sympathetic understanding of the client's "message"). Excellent listening and empathic skills.""",

    "Interpersonal Effectiveness": """
Severely Deficient: Therapist had poor interpersonal skills. Seemed hostile, demeaning, or in some other way destructive to the patient.
Poor: Therapist did not seem destructive, but had significant interpersonal problems. At times, therapist appeared unnecessarily impatient, aloof, insincere or had difficulty conveying confidence and competence.
Good: Therapist displayed a satisfactory degree of warmth, concern, confidence, genuineness, and professionalism. No significant interpersonal problems.
Outstanding: Therapist displayed optimal levels of warmth, concern, confidence, genuineness, and professionalism, appropriate for this particular patient in this session.""",

    "Collaboration": """
Severely Deficient: Therapist did not attempt to set up a collaboration with patient.
Poor: Therapist attempted to collaborate with patient, but had difficulty either defining a problem that the patient considered important or establishing rapport.
Good: Therapist was able to collaborate with patient, focus on a problem that both patient and therapist considered important, and establish rapport.
Outstanding: Collaboration seemed excellent; therapist encouraged patient as much as possible to take an active role during the session (e.g., by offering choices) so they could function as a "team".""",

    "Pacing and Efficient Use of Time": """
Severely Deficient: Therapist made no attempt to structure therapy time. Session seemed aimless.
Poor: Session had some direction, but the therapist had significant problems with structuring or pacing (e.g., too little structure, inflexible about structure, too slowly paced, too rapidly paced).
Good: Therapist was reasonably successful at using time efficiently. Therapist maintained appropriate control over flow of discussion and pacing.
Outstanding: Therapist used time efficiently by tactfully limiting peripheral and unproductive discussion and by pacing the session as rapidly as was appropriate for the patient.""",

    "Guided Discovery": """
Severely Deficient: Therapist relied primarily on debate, persuasion, or "lecturing." Therapist seemed to be "cross-examining" patient, putting the patient on the defensive, or forcing his/her point of view on the patient.
Poor: Therapist relied too heavily on persuasion and debate, rather than guided discovery. However, therapist's style was supportive enough that patient did not seem to feel attacked or defensive.
Good: Therapist, for the most part, helped patient see new perspectives through guided discovery (e.g., examining evidence, considering alternatives, weighing advantages and disadvantages) rather than through debate. Used questioning appropriately.
Outstanding: Therapist was especially adept at using guided discovery during the session to explore problems and help patient draw his/her own conclusions. Achieved an excellent balance between skillful questioning and other modes of intervention.""",

    "Focusing on Key Cognitions or Behaviors": """
Severely Deficient: Therapist did not attempt to elicit specific thoughts, assumptions, images, meanings, or behaviors.
Poor: Therapist used appropriate techniques to elicit cognitions or behaviors; however, therapist had difficulty finding a focus or focused on cognitions/behaviors that were irrelevant to the patient's key problems.
Good: Therapist focused on specific cognitions or behaviors relevant to the target problem. However, therapist could have focused on more central cognitions or behaviors that offered greater promise for progress.
Outstanding: Therapist very skillfully focused on key thoughts, assumptions, behaviors, etc. that were most relevant to the problem area and offered considerable promise for progress.""",

    "Strategy for Change": """
Severely Deficient: Therapist did not select cognitive-behavioral techniques.
Poor: Therapist selected cognitive-behavioral techniques; however, either the overall strategy for bringing about change seemed vague or did not seem promising in helping the patient.
Good: Therapist seemed to have a generally coherent strategy for change that showed reasonable promise and incorporated cognitive-behavioral techniques.
Outstanding: Therapist followed a consistent strategy for change that seemed very promising and incorporated the most appropriate cognitive-behavioral techniques.""",

    "Application of Cognitive-Behavioral Techniques": """
Severely Deficient: Therapist did not apply any cognitive-behavioral techniques.
Poor: Therapist used cognitive-behavioral techniques, but there were significant flaws in the way they were applied.
Good: Therapist applied cognitive-behavioral techniques with moderate skill.
Outstanding: Therapist very skillfully and resourcefully employed cognitive-behavioral techniques.""",

    "Homework": """
Severely Deficient: Therapist did not attempt to incorporate homework relevant to cognitive therapy.
Poor: Therapist had significant difficulties incorporating homework (e.g., did not review previous homework, did not explain homework in sufficient detail, assigned inappropriate homework).
Good: Therapist reviewed previous homework and assigned "standard" cognitive therapy homework generally relevant to issues dealt with in session. Homework was explained in sufficient detail.
Outstanding: Therapist reviewed previous homework and carefully assigned homework drawn from cognitive therapy for the coming week. Assignment seemed "custom tailored" to help patient incorporate new perspectives, test hypotheses, experiment with new behaviors discussed during session, etc."""
}

rate_limiter = RateLimiter(REQUESTS_PER_MINUTE)
request_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

def word_to_number(word: str) -> Optional[int]:
    """Convert word score to numeric score, handling N/A cases"""
    if isinstance(word, (int, float)):
        return word
    
    # Handle N/A cases (case insensitive)
    if isinstance(word, str) and re.match(r'^(n/a|not\s+applicable)$', word.lower().strip()):
        return min(WORD_TO_NUMBER.values())  # Return lowest score
        
    return LOWER_WORD_TO_NUMBER.get(word.lower(), None)

def number_to_word(number: int) -> Optional[str]:
    """Convert numeric score to word score"""
    if isinstance(number, str):
        return number
    return NUMBER_TO_WORD.get(number, None)

def load_transcripts(DATA_DIR: str) -> List[Dict[str, str]]:
    """Load transcripts from the specified directory"""
    transcripts = []
    group_dirs = [name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))]
    for group_dir in group_dirs:
        group_path = os.path.join(DATA_DIR, group_dir)
        for root, _, files in os.walk(group_path):
            for file in files:
                if file.endswith('.txt'):
                    try:
                        transcript_id = '_'.join(file.split('_')[1:]).rsplit('.', 1)[0]
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            content = f.read()
                        transcripts.append({"id": transcript_id, "content": content, "group": group_dir})
                    except Exception as e:
                        logger.warning(f"Error reading file {file}: {str(e)}")
    logger.info(f"Loaded {len(transcripts)} transcripts from {DATA_DIR}")
    return transcripts

async def vertex_evaluate(transcript: str, retry: int = 10, backoff_factor: float = 5) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Evaluate transcript using Vertex AI for all categories at once"""
    
    prompt = f"""As a preeminent CBT evaluation expert with 30+ years of experience training and supervising thousands of therapists internationally, conduct an exhaustively detailed and uncompromisingly critical analysis of the following therapy session transcript. You must maintain the absolute highest standards of CBT practice. Your evaluation should be particularly attuned to subtle deficiencies that could impact therapeutic outcomes.

    For each category defined in the CTRS criteria: {json.dumps(CTRS_CRITERIA, indent=2)}

    Provide an extraordinarily thorough evaluation including:

    1. A score using this deliberately stringent rating scale:
    - Severely Deficient (Complete absence of skill/awareness, potentially harmful)
    - Very Poor (Major systematic failures requiring immediate attention)
    - Poor (Significant pervasive deficits across multiple areas)
    - Unsatisfactory (Notable weaknesses outweigh strengths)
    - Below Average (Some basic competency but clear limitations)
    - Adequate (Meets minimal standards but room for improvement)
    - Good (Solid performance with minor areas for growth)
    - Very Good (Strong skills demonstrated consistently)
    - Excellent (Superior performance with minimal flaws)
    - Outstanding (Exceptional mastery rarely achieved)

    2. An exhaustive justification that MUST include:
    a) Specific examples from the transcript demonstrating both strengths and deficiencies
    b) Detailed analysis of missed opportunities and suboptimal interventions
    c) Thorough examination of both obvious and subtle weaknesses
    d) Concrete, actionable improvements needed for each identified issue
    e) Explicit references to CTRS criteria and current best practices
    f) Analysis of potential downstream impacts on therapeutic outcomes
    g) Comparison to exemplary CBT implementation standards

    3. Additional Critical Considerations:
    - Examine cultural competency and therapeutic alliance factors
    - Analyze appropriateness of intervention selection
    - Evaluate scaffolding of interventions and skill development

    IMPORTANT: Format your response exactly as shown below with no additional text before or after:
    {{
        "Agenda": {{"score": "", "justification": ""}},
        "Feedback": {{"score": "", "justification": ""}},
        "Understanding": {{"score": "", "justification": ""}},
        "Interpersonal Effectiveness": {{"score": "", "justification": ""}},
        "Collaboration": {{"score": "", "justification": ""}},
        "Pacing and Efficient Use of Time": {{"score": "", "justification": ""}},
        "Guided Discovery": {{"score": "", "justification": ""}},
        "Focusing on Key Cognitions or Behaviors": {{"score": "", "justification": ""}},
        "Strategy for Change": {{"score": "", "justification": ""}},
        "Application of Cognitive-Behavioral Techniques": {{"score": "", "justification": ""}},
        "Homework": {{"score": "", "justification": ""}}
    }}

    Transcript:
    {transcript}
    """
    
    attempt = 0
    while attempt < retry:
        try:
            async with request_semaphore:
                await rate_limiter.acquire()
                
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                content = response.text.strip()
                scores = {}
                justifications = {}
                invalid_categories = []
                
                for category in CATEGORIES:
                    # Extract score using regex
                    score_pattern = f'"{category}"\\s*:\\s*{{\\s*"score"\\s*:\\s*"([^"]*)"'
                    score_match = re.search(score_pattern, content)
                    if not score_match:
                        logger.warning(f"Score not found for category: {category}")
                        invalid_categories.append(category)
                        continue
                    score = score_match.group(1)
                    
                    # Extract justification using regex
                    justification_pattern = f'"{category}"\\s*:\\s*{{[^}}]*"justification"\\s*:\\s*"([^"]*)"'
                    justification_match = re.search(justification_pattern, content)
                    if not justification_match:
                        logger.warning(f"Justification not found for category: {category}")
                        invalid_categories.append(category)
                        continue
                    justification = justification_match.group(1)
                    
                    score_numeric = word_to_number(score)
                    if score_numeric is not None:
                        scores[category] = score_numeric
                        justifications[category] = justification
                    else:
                        invalid_categories.append(category)
                        logger.warning(f"Invalid score '{score}' for category '{category}'")
                
                # If there are invalid categories and we haven't exceeded retries
                if invalid_categories and attempt + 1 < retry:
                    logger.warning(f"Invalid ratings found for categories: {invalid_categories}. Retrying...")
                    attempt += 1
                    wait_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                    continue
                
                # If we have some valid scores or we've run out of retries
                if scores or attempt + 1 == retry:
                    # For any remaining invalid categories, assign lowest score
                    for category in invalid_categories:
                        scores[category] = min(WORD_TO_NUMBER.values())
                        justifications[category] = "Rating failed after multiple retries or N/A response"
                    
                    return scores, justifications

        except Exception as e:
            logger.error(f"Error during Vertex AI evaluation (attempt {attempt + 1}/{retry}): {str(e)}")
            if attempt + 1 < retry:
                wait_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Waiting {wait_time:.2f} seconds before retry...")
                await asyncio.sleep(wait_time)
            attempt += 1

    # If all retries failed, assign lowest scores
    logger.warning(f"Failed to get valid evaluation after {retry} attempts. Assigning lowest scores.")
    return {category: min(WORD_TO_NUMBER.values()) for category in CATEGORIES}, {category: "Evaluation failed after multiple retries" for category in CATEGORIES}

async def process_single_transcript(transcript: Dict[str, str], group_name: str, results: List[Dict[str, str]]):
    """Process a single transcript and add results to the results list"""
    transcript_id = transcript.get("id", "unknown_id")
    content = transcript.get("content", "")
    
    if not content:
        logger.warning(f"Transcript ID {transcript_id} has empty content. Skipping.")
        return
    
    try:
        scores, justifications = await vertex_evaluate(content)
        if scores:
            result = {
                "id": transcript_id,
                "group": group_name,
                **scores,
                "justifications": json.dumps(justifications)
            }
            results.append(result)
            logger.debug(f"Evaluated transcript ID {transcript_id}.")
        else:
            logger.warning(f"Transcript ID {transcript_id} failed evaluation.")
    except Exception as e:
        logger.error(f"Error processing transcript ID {transcript_id}: {str(e)}")

async def evaluate_transcripts(transcripts: List[Dict[str, str]], group_name: str) -> pd.DataFrame:
    results = []
    temp_file = os.path.join(EVALUATION_RESULTS_DIR, f"{group_name}_scores_temp.csv")
    
    if os.path.exists(temp_file):
        existing_df = pd.read_csv(temp_file)
        processed_ids = set(existing_df['id'].values)
        results = existing_df.to_dict('records')
        transcripts = [t for t in transcripts if t['id'] not in processed_ids]
        logger.info(f"Resuming evaluation for group '{group_name}' with {len(transcripts)} remaining transcripts.")
    
    chunk_size = MAX_CONCURRENT_REQUESTS
    for i in range(0, len(transcripts), chunk_size):
        chunk = transcripts[i:i + chunk_size]
        tasks = [process_single_transcript(transcript, group_name, results) for transcript in chunk]
        await asyncio.gather(*tasks)
        
        if results:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(temp_file, index=False)
            logger.info(f"Saved intermediate results for group '{group_name}' after processing {len(results)} transcripts.")
        
        if i + chunk_size < len(transcripts):
            await asyncio.sleep(1)

    final_df = pd.DataFrame(results)
    
    final_path = os.path.join(EVALUATION_RESULTS_DIR, f"{group_name}_scores.csv")
    final_df.to_csv(final_path, index=False)
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    logger.info(f"Completed evaluation of {len(final_df)} transcripts for group '{group_name}'.")
    return final_df

async def main():
    logger.info("Starting transcript evaluation and analysis.")

    try:
        transcripts = load_transcripts(DATA_DIR)
        logger.info(f"Loaded a total of {len(transcripts)} transcripts.")
        
        progress_file = os.path.join(EVALUATION_RESULTS_DIR, "evaluation_progress.json")
        progress = {}
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        
        if len(transcripts) > REQUESTS_PER_MINUTE:
            logger.info("Large number of transcripts detected. Adding initial delay to prevent rate limiting.")
            await asyncio.sleep(2)
            
    except FileNotFoundError as e:
        logger.error(f"Transcript directory not found: {str(e)}")
        return
    except Exception as e:
        logger.error(f"Unexpected error loading transcripts: {str(e)}")
        return

    grouped_transcripts = defaultdict(list)
    for transcript in transcripts:
        group = transcript['group']
        grouped_transcripts[group].append(transcript)

    all_scores_dfs = []
    
    for group_name, group_transcripts in grouped_transcripts.items():
        if group_name in progress and progress[group_name] == "completed":
            logger.info(f"Skipping already processed group: {group_name}")
            group_scores_path = os.path.join(EVALUATION_RESULTS_DIR, f"{group_name}_scores.csv")
            if os.path.exists(group_scores_path):
                df = pd.read_csv(group_scores_path)
                all_scores_dfs.append(df)
            continue

        logger.info(f"Processing group: {group_name}")
        try:
            df = await evaluate_transcripts(group_transcripts, group_name)
            all_scores_dfs.append(df)
            
            progress[group_name] = "completed"
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
            
            if len(group_transcripts) > REQUESTS_PER_MINUTE / 2:
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"Error processing group '{group_name}': {str(e)}")
            progress[group_name] = "failed"
            with open(progress_file, 'w') as f:
                json.dump(progress, f)

    if all_scores_dfs:
        all_scores_df = pd.concat(all_scores_dfs, ignore_index=True)
        combined_scores_path = os.path.join(EVALUATION_RESULTS_DIR, "all_scores.csv")
        all_scores_df.to_csv(combined_scores_path, index=False)
        logger.info(f"Saved combined evaluation results to '{combined_scores_path}'.")

    logger.info("Transcript evaluation and analysis completed successfully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error during execution: {str(e)}")
        raise