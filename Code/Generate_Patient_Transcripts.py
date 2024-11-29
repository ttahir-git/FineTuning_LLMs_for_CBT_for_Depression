# Import necessary libraries
import random
import json
import os
import time
from typing import List, Dict
import logging
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
from os import getenv
import re

# Configuration constants
NUM_DEPRESSION_SETS = 100
NUM_SESSIONS = 20  
THERAPIST_NAME = "Nova"
OUTPUT_DIR = "depression_cbt_transcripts_405b_v11"
SESSION_DURATION_MINUTES = 75
API_CALL_DELAY = 1  
MAX_RETRIES = 20 
MAX_REGENERATION_ATTEMPTS = 20 

# Set up logging
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_filename = f"Depression_CBT_Generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(log_directory, log_filename)

# Custom filter to exclude certain log messages
class NoRequestFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith(
            ("HTTP Request:", "connect_tcp", "start_tls", "send_request")
        )

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_path, mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.addFilter(NoRequestFilter())

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# OpenRouter API configuration
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

# Constants for patient profile generation
AGE_RANGES = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
GENDERS = ["Male", "Female", "Non-binary", "Transgender", "Other"]
ETHNICITIES = ["Caucasian", "African American", "Hispanic", "South Asian", "Middle Eastern", "East Asian"]
EDUCATION_LEVELS = ["High School", "Some College", "Bachelor's Degree", "Master's Degree", "Doctoral Degree"]
OCCUPATIONS = ["Student", "Unemployed", "Blue-collar worker", "White-collar worker", "Self-employed", "Retired"]
SYMPTOM_SEVERITIES = ["Mild", "Moderate", "Severe"]
ENGAGEMENT_LEVELS = ["Low", "Medium", "High"]

LIFE_EVENTS = [
    "Recent job loss", "Divorce or relationship breakup", "Death of a loved one",
    "Chronic illness diagnosis", "Financial difficulties", "Moving to a new city",
    "Becoming a parent", "Empty nest syndrome", "Retirement", "Career change",
    "Traumatic event (e.g., accident, assault)", "Academic struggles",
    "Workplace stress or burnout", "Caring for an ill family member",
    "Identity crisis", "Substance abuse recovery", "Military service or deployment",
    "Cultural adjustment (e.g., immigration)", "Coming out as LGBTQ+",
    "Infertility or pregnancy loss"
]

FAMILY_BACKGROUNDS = [
    "Supportive and close-knit", "Dysfunctional with history of abuse",
    "Single-parent household", "Adopted or foster care background",
    "Blended family", "Estranged from family", "First-generation immigrant",
    "Multi-generational household", "Only child", "Large family with many siblings",
    "Family history of mental illness", "High-achieving family with pressure to succeed",
    "Family affected by addiction", "Religious or conservative upbringing",
    "LGBTQ+ family", "Military family with frequent relocations",
    "Family with financial struggles", "Family with chronic illness",
    "Culturally diverse family", "Family affected by divorce or separation"
]

HOBBIES_INTERESTS = [
    "Reading", "Writing", "Painting or drawing", "Photography", "Gardening",
    "Cooking or baking", "Playing musical instruments", "Singing", "Dancing",
    "Yoga or meditation", "Hiking or camping", "Traveling", "Sports (specify)",
    "Video games", "Board games", "Volunteering", "Animal care or pet ownership",
    "Collecting (specify)", "DIY projects", "Learning languages"
]

SOCIAL_SUPPORT_SYSTEMS = [
    "Strong network of friends", "Limited social connections",
    "Supportive partner or spouse", "Active in community groups",
    "Relies mainly on family for support", "Professional support network",
    "Online support communities", "Religious or spiritual community",
    "Support group for specific issues", "Therapy or counseling",
    "Mentorship relationships", "Work colleagues", "Neighborhood connections",
    "Sports team or fitness group", "Creative or artistic community",
    "LGBTQ+ support network", "Cultural or ethnic community",
    "Veterans support group", "Single parents support group",
    "Chronic illness support network"
]

PERSONALITY_TRAITS = [
    "Introverted", "Extroverted", "Anxious", "Confident", "Pessimistic", "Optimistic",
    "Perfectionist", "Easy-going", "Analytical", "Creative", "Sensitive",
    "Resilient", "Impulsive", "Cautious", "Ambitious", "Laid-back",
    "Empathetic", "Reserved", "Adventurous", "Detail-oriented"
]

COPING_MECHANISMS = [
    "Avoidance", "Substance use", "Overworking", "Social withdrawal",
    "Exercise", "Creative expression", "Mindfulness or meditation",
    "Seeking social support", "Problem-solving", "Emotional eating",
    "Journaling", "Self-harm", "Therapy or counseling", "Religious or spiritual practices",
    "Humor or laughter", "Retail therapy", "Excessive sleeping", "Anger or aggression",
    "Perfectionism", "Helping others"
]

# Updated CBT Phases
CBT_PHASES = {
    "Assessment": (1, 3),
    "Initial": (4, 7),
    "Middle": (8, 17),
    "Termination": (18, 20)
}

# CBT Techniques
CBT_TECHNIQUES = {
   "Assessment": [
        "Assessment and Diagnosis - Identify specific functional deficits in various life domains (work, relationships, self-care). Create a detailed timeline of depression onset, course, and any previous treatment experiences.",
        "Psychoeducation - Introduce the cognitive model and CBT rationale. Explain the cognitive conceptualization framework: Situation/Trigger → Automatic Thoughts → Emotions → Behaviors ← Intermediate Beliefs ← Core Beliefs. Set expectations for therapy, including the collaborative nature of CBT and the importance of active participation.",
        "Emotion Identification - Teach the patient to identify and label emotions, distinguishing between different emotions and their intensities.",
        "Automatic Thought Identification - Teach the concept of 'hot thoughts' - those most strongly linked to emotional reactions.",
        "Goal Setting - Discuss and establish specific goals for the course of therapy."
        "Homework Assignment - Assign homework related to automatic thought and emotion identification in various situations."
    ],
    "Initial": [
        "Distress Tolerance Techniques - Introduce techniques such as mindfulness, grounding exercises, and deep breathing to manage overwhelming emotions. Teach progressive muscle relaxation and guided imagery for stress reduction.",
        "Behavioral Activation - Explain the role of avoidance in maintaining depression. Encourage the patient to engage in pleasant activities and monitor their mood to counteract avoidance and withdrawal. Use activity scheduling to plan and track these activities. Implement graded task assignments to build a sense of mastery and pleasure.",
        "Thought Records - Columns typically include: Date/Time, Situation/Trigger, Emotions (and intensity), Automatic Thoughts (Identify the most prominent maladaptive automatic thought), Evidence For/Against the thought identified, Alternative Thought (reframing of the original thought), and Outcome (re-rate emotion).",
        "Homework Assignment - Assign homework, such as completing thought records, engaging in planned activities, and practicing distress tolerance techniques. Emphasize the importance of homework in making progress and generalizing skills."
    ],
    "Middle": [
        "Thought Records - Columns typically include: Date/Time, Situation/Trigger, Emotions (and intensity), Automatic Thoughts (Identify the most prominent maladaptive automatic thought), Evidence For/Against the thought identified, Alternative Thought (reframing of the original thought), and Outcome (re-rate emotion).",
        "Cognitive Distortions",
        "Downward Arrow Technique - Use this technique to uncover underlying intermediate and core beliefs (e.g., repeatedly asking 'If that were true, what would that mean to you?''). Help patients connect surface-level thoughts to deeper beliefs and attitudes.",
        "Identifying Intermediate and Core Beliefs - Help the patient identify their unique intermediate (if/then statements) and core beliefs.Explore the connection between situations, thoughts, and underlying beliefs.",
        "Cognitive and Behavioral Experiments - Collaboratively design experiments to test the validity of the patient's beliefs and gather evidence to challenge them. Create a graded hierarchy of experiments, gradually challenging more deeply held beliefs. Use prediction logs to track expectations and outcomes of experiments.",
        "Advanced Cognitive Restructuring - Continue working on thought records, using Socratic questioning to challenge negative thoughts. Teach the patient to evaluate the evidence for and against their negative thoughts.",
        "Reframing Core Beliefs - Help the patient develop more adaptive and balanced core beliefs. Use techniques such as the Positive Data Log to accumulate evidence for new beliefs. Practice generating alternative explanations for events that reinforce negative core beliefs.",
        "Addressing Specific Depressive Symptoms - Target rumination through techniques like worry time and thought stopping. Address perfectionism by challenging unrealistic standards and promoting self-compassion. Work on improving assertiveness and interpersonal effectiveness to address social withdrawal.",
        "Expanded Homework - Assign cognitive and behavioral experiments as homework, along with thought records and behavioral activation tasks. Encourage journaling to track mood patterns and identify triggers."
    ],
    "Termination": [
        "Reinforce Learning - Review the progress made and the skills learned during therapy. Highlight the patient's accomplishments and the improvements in their mood and functioning. Create a personalized 'toolbox' of CBT skills and strategies.",
        "Build Self-Efficacy - Encourage the patient to take credit for their progress and instill confidence in their ability to continue using CBT skills independently. Practice problem-solving future challenges using CBT techniques.",
        "Relapse Prevention Planning - Develop a plan to maintain gains and manage potential setbacks, including identifying early warning signs of depression and strategies to cope with them. Create a 'wellness action plan' that outlines steps to take if symptoms begin to worsen.",
        "Role Reversal - Have the patient teach you about CBT concepts and skills to assess their understanding and consolidate their knowledge.Encourage the patient to 'become their own therapist' by applying CBT principles independently.",
        "Meta-Thought Records - Complete thought records about the experience of CBT itself and the progress made. Address any concerns or anxieties about ending therapy.",
        "Coping Cards - Create coping cards with reminders of adaptive thoughts, strategies, and new core beliefs to support the patient after therapy. Develop personalized affirmations and mantras to reinforce positive beliefs.",
        "Support Planning - Discuss how the patient will support themselves after therapy, including accessing social support, engaging in self-care, and seeking professional help if needed. Explore options for ongoing support, such as support groups or periodic 'booster' sessions.",
        "Gradual Termination - Gradually taper the frequency of sessions to ease the transition to independence and provide opportunities for the patient to practice their skills. Conduct a final review of progress, celebrating achievements and acknowledging ongoing areas for growth."
    ]
}

# Ending phrases for detecting premature wrap-ups
ENDING_PHRASES = [
    r"let's summarize what we covered",
    r"let's summarize what we've covered",
    r"let me summarize the main points",
    r"let's summarize the main points",
    r"I want to summarize what we covered",
    r"I'd like to wrap up by summarizing",
    r"let's summarize what we've discussed",
    r"Before we wrap up, let's summarize",
    r"provide a summary",
    r"Let's wrap up today's session by summarizing",
    r"As we wrap up today's session, let's summarize",
    r"Let's take a moment to summarize our session",
    r"Before we wrap up today, let's summarize",
    r"As we wrap up today, let's summarize",
    r"wrap up",
    r"wrap-up",
    r"Before we end",
    r"goodbye",
    r"farewell",
    r"bye",
    r"see you next time",
    r"until next session",
    r"take care",
    r"That concludes our session",
    r"Our time is almost up",
    r"We're nearing the end of our session",
    r"As we come to the close of our session",
    r"Before we finish for today",
    r"To conclude our session",
]

# Compile the ending phrases into a single regex pattern
ENDING_PATTERN = re.compile("|".join(ENDING_PHRASES), re.IGNORECASE)

def rate_limited_api_call(
    messages: List[Dict[str, str]], 
    max_tokens: int = 4096, 
    temperature: float = 0.7
) -> Dict:
    """
    Makes an API call to the OpenRouter API with rate limiting and error handling.

    Args:
    messages (List[Dict[str, str]]): The messages to send to the API.
    max_tokens (int): The maximum number of tokens to generate.
    temperature (float): The temperature to use for generation.

    Returns:
    Dict: The API response.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="nousresearch/hermes-3-llama-3.1-405b",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if not response.choices or not response.choices[0].message:
                raise ValueError("Invalid response structure from OpenRouter API")

            time.sleep(API_CALL_DELAY)  # Delay between API calls
            return response
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                wait_time = (2 ** attempt) + random.random()  # Exponential backoff
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to generate response after {MAX_RETRIES} attempts")
                raise

def generate_patient_characteristics() -> Dict[str, str]:
    """
    Generates a set of random patient characteristics.

    Returns:
    Dict[str, str]: A dictionary of patient characteristics.
    """
    return {
        "Age": random.choice(AGE_RANGES),
        "Gender": random.choice(GENDERS),
        "Ethnicity": random.choice(ETHNICITIES),
        "Education": random.choice(EDUCATION_LEVELS),
        "Occupation": random.choice(OCCUPATIONS),
        "Symptom_Severity": random.choice(SYMPTOM_SEVERITIES),
        "Engagement_Level": random.choice(ENGAGEMENT_LEVELS),
        "Life_Events": random.sample(LIFE_EVENTS, k=random.randint(1, 3)),
        "Family_Background": random.choice(FAMILY_BACKGROUNDS),
        "Hobbies_Interests": random.sample(HOBBIES_INTERESTS, k=random.randint(2, 5)),
        "Social_Support": random.choice(SOCIAL_SUPPORT_SYSTEMS),
        "Personality_Traits": random.sample(PERSONALITY_TRAITS, k=random.randint(2, 4)),
        "Coping_Mechanisms": random.sample(COPING_MECHANISMS, k=random.randint(2, 4))
    }

def generate_patient_profile(characteristics: Dict[str, str]) -> str:
    """
    Generates a detailed patient profile for depression based on given characteristics.

    Args:
    characteristics (Dict[str, str]): Predetermined patient characteristics.

    Returns:
    str: A detailed patient profile.
    """
    prompt = f"""Create a detailed and unique patient profile for a person with depression, based on the following characteristics:

    {json.dumps(characteristics, indent=2)}

    Include:
    1. A vivid and realistic backstory that incorporates the given characteristics, life events, and family background.
    2. Detailed description of the patient's current life situation, including work, relationships, and daily challenges.
    3. Specific depressive symptoms and their impact on the patient's life, considering their unique circumstances.
    4. Onset and progression of depression, relating it to the patient's life events and experiences.
    5. Previous treatment history, if any, and the patient's attitude towards therapy.
    6. Current coping mechanisms and their effectiveness, considering the patient's personality traits.
    7. Detailed description of the patient's social support system and how they utilize it.
    8. Personal strengths, resources, and resilience factors.
    9. Potential challenges in therapy, considering the patient's background and personality.
    10. Specific, personalized goals for treatment that align with the patient's life circumstances and values.
    11. Any cultural, religious, or personal beliefs that may influence their experience of depression or approach to treatment.

    The profile should be comprehensive, realistic, and highly individualized, providing a clear picture of a unique individual's current situation, challenges, and potential for growth in therapy. Ensure that the backstory and life experiences are diverse and distinct from other profiles, while maintaining believability and relevance to the patient's depression."""

    messages = [{"role": "user", "content": prompt}]
    response = rate_limited_api_call(messages)
    return response.choices[0].message.content

def check_for_session_summary(transcript: str) -> bool:
    """
    Check if the transcript contains a 'Session Summary' section.

    Args:
    transcript (str): The generated transcript.

    Returns:
    bool: True if 'Session Summary' is found, False otherwise.
    """
    return 'Session Summary:' in transcript

def check_for_agenda(transcript: str) -> bool:
    """
    Check if the transcript contains an agenda setting.

    Args:
    transcript (str): The generated transcript.

    Returns:
    bool: True if agenda setting is found, False otherwise.
    """
    return any(keyword in transcript.lower() for keyword in ['agenda', 'plan for today', 'focus for today'])


def get_cbt_phase(session_number: int) -> str:
    """
    Determines the CBT phase based on the session number.

    Args:
    session_number (int): The current session number.

    Returns:
    str: The current CBT phase ("Assessment", "Initial", "Middle", or "Termination").
    """
    for phase, (start, end) in CBT_PHASES.items():
        if start <= session_number <= end:
            return phase
    return "Unknown"

def get_techniques_for_phase(phase: str) -> List[str]:
    """
    Returns all CBT techniques for a given phase.

    Args:
    phase (str): The CBT phase ("Assessment", "Initial", "Middle", or "Termination").

    Returns:
    List[str]: A list of all techniques for the given phase.
    """
    return CBT_TECHNIQUES[phase]

def check_premature_wrap_up(transcript: str, part: int) -> bool:
    """
    Check if the generated transcript part contains premature wrap-up phrases.

    Args:
    transcript (str): The generated transcript part.
    part (int): The current part number (1, 2, 3, or 4).

    Returns:
    bool: True if a premature wrap-up is detected, False otherwise.
    """
    if part < 4 and ENDING_PATTERN.search(transcript):
        return True
    return False

def check_for_asterisk(transcript: str) -> bool:
    """
    Check if the transcript contains any asterisks.

    Args:
    transcript (str): The generated transcript.

    Returns:
    bool: True if an asterisk is found, False otherwise.
    """
    return '*' in transcript

def check_double_messages(transcript: str) -> bool:
    """
    Checks if there are any double messages from either the therapist or patient.

    Args:
    transcript (str): The generated transcript.

    Returns:
    bool: True if double messages are detected, False otherwise.
    """
    lines = transcript.split('\n')
    previous_speaker = None
    for line in lines:
        if line.startswith("Therapist:") or line.startswith("Patient:"):
            current_speaker = line.split(':')[0]
            if current_speaker == previous_speaker:
                return True
            previous_speaker = current_speaker
    return False

def extract_session_summary(transcript: str) -> str:
    """
    Extracts the session summary from the transcript.
    """
    summary_match = re.search(r"Session Summary:(.*?)(?=\n\nTherapist:|$)", transcript, re.DOTALL | re.IGNORECASE)
    if summary_match:
        return summary_match.group(1).strip()
    return ""


def get_part_specific_instructions(part: int) -> str:
    """
    Returns the specific instructions for each part of the CBT session.

    Args:
    part (int): The part number (1, 2, 3, or 4).

    Returns:
    str: The specific instructions for the given part.
    """
    instructions = {
        1: """For part 1, include:
    1. Respond to the patient's introductory message and summary of the last session and continue generating the rest of the transcript using the guidance below.
    2. The AI therapist should acknowledge its limitations as an AI including: 
        - Inability to access real-time information or personal details beyond what's provided in the session
        - Potential for generating inaccurate information (hallucination)
        - Inability to provide crisis intervention or emergency services
        - Lack of ability to directly observe or interpret non-verbal cues 
    3. A thorough check-in, exploring the patient's mood, experiences, and any changes since the last session.
    4. Setting a detailed agenda for the session, with the therapist and patient collaborating on priorities.
    """,

        2: """For part 2, include:
    1. In-depth discussion and review of homework (if assigned in the previous session), including challenges faced and insights gained. DO NOT RUSH THIS, explore it deeply with the patient. 
    2. Begin application of CBT techniques and exercises appropriate for the current phase, focusing on the techniques listed for this phase.
    3. Addressing any resistance or skepticism from the patient in a compassionate manner.
    4. DO NOT say any goodbye to the patient or 'wrap-up' in this part of the session as we are still in the middle of the session.""",

        3: """For part 3, include:
    1. Continued application of CBT techniques, going into greater depth and detail.
    2. Exploring connections between thoughts, feelings, and behaviors related to the patient's specific experiences.
    3. Practicing skills or techniques introduced in the session, with the therapist providing guidance and feedback.
    4. DO NOT say any goodbye to the patient or 'wrap-up' in this part of the session as we are still in the middle of the session.""",

        4: """For part 4, include:
    1. Continue and wrap up application of CBT techniques. 
    2. A brief check-in on the patient's current mood and any final thoughts.
    3. Collaborative discussion and assignment of homework. 
    4. IMPORTANT: At the very end of part 4, the therapist should provide a VERY VERY DETAILED "Session Summary" titled "Session Summary:" that includes:
       - Main topics discussed
       - Techniques or exercises used (with details on progress and plans for the future i.e. building on or introducing new techniques). This should include a list of ALL techniques not just the ones for this session, even the ones from previous sessions to ensure that they are followed up on appropriately.
       - Homework assigned [in full detail]
       - Patient's progress, challenges, depression symptoms and important personal details
       - Core beliefs and intermediate beliefs [IF DISCUSSED in this or previous sessions]
       - Any important information to carry forward to future sessions (this is very important)
       - Plan for the next session. """
    }

    return instructions.get(part, "")

def get_assessment_part_instructions(session_number: int, part: int) -> str:
    """
    Returns specific instructions for each part of assessment phase sessions (1-3).
    
    Args:
    session_number (int): The current session number (1, 2, or 3).
    part (int): The current part number (1, 2, 3, or 4).
    
    Returns:
    str: Specific instructions for the assessment phase session part.
    """
    instructions = {
        1: {
            1: """
            - The patient should introduce themselves to the therapist as this is their first session. 
            - The patient MUST speak first in the session. The first message of the session has to be from the patient NOT from the therapist. 
            - Introduce yourself as the AI therapist and explain the purpose of the assessment phase.
            - The AI therapist should acknowledge its limitations as an AI including: 
                - Inability to access real-time information or personal details beyond what's provided in the session
                - Potential for generating inaccurate information (hallucination)
                - Inability to provide crisis intervention or emergency services
                - Lack of ability to directly observe or interpret non-verbal cues 
            - Begin gathering information about the patient's reasons for seeking therapy.
            - Start exploring the patient's current life situation and challenges.
            - Do not conclude the session or introduce CBT concepts yet.
            """,
            2: """
            - Continue gathering information about the patient's depression symptoms and their impact.
            - Explore the patient's personal history, focusing on factors that may contribute to their current state.
            - Begin to identify patterns in the patient's thoughts, feelings, and behaviors.
            - Do not conclude the session or introduce CBT concepts yet.
            """,
            3: """
            - Delve deeper into the patient's coping mechanisms and support systems.
            - Explore any previous experiences with therapy or treatment for depression.
            - Begin to formulate a preliminary understanding of the patient's situation.
            - Do not conclude the session or introduce CBT concepts yet.
            """,
            4: """
            - Briefly introduce the concept of CBT and how it might help with depression.
            - Set expectations for the next session.
            - Collaborative discussion and assignment of homeowork. 
            - IMPORTANT: At the end of this part therapist should provide a VERY VERY DETAILED "Session Summary" titled "Session Summary:" that includes:
                - Main topics discussed
                - Techniques or exercises used (with details on progress and plans for the future i.e. building on or introducting new techniques)
                - Homework assigned [in full detail]
                - Patient's progress, challenges, depression symptoms and important personal details 
                - Any important information to carry forward to future sessions (this is very important)
                - Plan for the next session.             
            """
        },
        2: {
            1: """
            - Respond to the patient's introductory message and summary of the last session and continue generating the rest of the transcript using the guidance below.
            - The AI therapist should acknowledge its limitations as an AI including: 
                - Inability to access real-time information or personal details beyond what's provided in the session
                - Potential for generating inaccurate information (hallucination)
                - Inability to provide crisis intervention or emergency services
                - Lack of ability to directly observe or interpret non-verbal cues 
            - REVIEW THE HOMEWORK in detail from the previous session.
            - Begin introducing the CBT framework, focusing on the connection between thoughts, feelings, and behaviors.
            - Start exploring the patient's specific thought patterns related to their depression.
            - Do not conclude the session.
            """,
            2: """
            - Continue explaining the CBT model, using examples from the patient's experiences.
            - Introduce the concept of cognitive distortions and help the patient identify some of their own.
            - Begin discussing how CBT techniques might apply to the patient's situation.
            - Do not conclude the session.
            """,
            3: """
            - Explore the patient's reactions to the CBT concepts introduced so far.
            - Begin discussing potential therapy goals, but do not finalize them.
            - Introduce the idea of homework in CBT and its importance.
            - Do not conclude the session.
            """,
            4: """
            - Review the CBT concepts discussed in this session.
            - Collaboratively set a small, achievable homework task related to thought monitoring.
            - Collaborative discussion and assignment of homeowork. 
            - IMPORTANT: At the end of this part therapist should provide a VERY VERY DETAILED "Session Summary" titled "Session Summary:" that includes:
                - Main topics discussed
                - Techniques or exercises used (with details on progress and plans for the future i.e. building on or introducting new techniques)
                - Homework assigned [in full detail]
                - Patient's progress, challenges, depression symptoms and important personal details 
                - Any important information to carry forward to future sessions (this is very important)
                - Plan for the next session.
            """
        },
        3: {
            1: """
            - Respond to the patient's introductory message and summary of the last session and continue generating the rest of the transcript using the guidance below.
            - The AI therapist should acknowledge its limitations as an AI including: 
                - Inability to access real-time information or personal details beyond what's provided in the session
                - Potential for generating inaccurate information (hallucination)
                - Inability to provide crisis intervention or emergency services
                - Lack of ability to directly observe or interpret non-verbal cues 
            - REVIEW THE HOMEWORK in detail from the previous session.
            - Finalize the CBT conceptualization of the patient's depression.
            - Begin the process of setting specific, measurable goals for therapy.
            - Do not conclude the session.
            """,
            2: """
            - Continue the goal-setting process, ensuring goals are realistic and aligned with CBT principles.
            - Introduce basic CBT techniques that will be used in future sessions.
            - Begin practicing a simple CBT technique (e.g., thought recording).
            - Do not conclude the session.
            """,
            3: """
            - Discuss the patient's experience with the CBT technique introduced earlier.
            - Finalize therapy goals and create a tentative treatment plan.
            - Address any concerns or questions the patient has about the therapy process.
            - Do not conclude the session.
            """,
            4: """
            - Review the goals and treatment plan established in this session.
            - Collaborative discussion and assignment of homeowork. 
            - IMPORTANT: At the end of this part therapist should provide a VERY VERY DETAILED "Session Summary" titled "Session Summary:" that includes:
                - Main topics discussed
                - Techniques or exercises used (with details on progress and plans for the future i.e. building on or introducting new techniques)
                - Homework assigned [in full detail]
                - Patient's progress, challenges, depression symptoms and important personal details 
                - Any important information to carry forward to future sessions (this is very important)
                - Plan for the next session.
            """
        }
    }
    return instructions.get(session_number, {}).get(part, "")

def generate_assessment_session_part(
    set_number: int,
    session_number: int,
    patient_profile: str,
    previous_summary: str,
    part: int,
    previous_parts: List[str]
) -> Dict[str, str]:
    """
    Generates a part of an assessment phase CBT session transcript (sessions 1-3).
    
    Args:
    set_number (int): The current set number.
    session_number (int): The current session number (1, 2, or 3).
    patient_profile (str): The patient's profile.
    previous_summary (str): Summary of the previous session, if any.
    part (int): The current part of the session (1, 2, 3, or 4).
    previous_parts (List[str]): The content of the previous parts of the current session.
    
    Returns:
    Dict[str, str]: A dictionary containing the session transcript part and word count.
    """
    general_instructions = f"""Generate part {part} of 4 for a detailed and realistic transcript of CBT assessment session {session_number} of 3 for a patient with depression. 
    Therapist name: {THERAPIST_NAME}
    Patient profile: {patient_profile}
    
    Previous session summary: {previous_summary}
    
    Previous parts of this session:
    {"".join(previous_parts)}

    Important notes:
    1. This is part {part} of 4 of the current assessment session. Ensure continuity with previous parts and DO NOT conclude the session unless this is part 4.
    2. Use the format "Patient: " for the patient's dialogue and "Therapist: " for the therapist's dialogue.
    3. This is a text-based interaction between an AI CBT therapist (Nova) and a patient. Do not include any non-verbal cues, pauses, or actions.
    4. The AI therapist should provide thorough explanations with examples when introducing new concepts.
    5. Regularly check the patient's understanding and explore any questions.
    6. Check in regularly to assess the patient's emotional state at various points in the session. 
    7. Incorporate the patient's specific background, goals, and challenges from their profile into the session content.
    8. The output should ONLY be the transcript dialogue, with no additional descriptions or explanations.
    9. If the patient expresses thoughts of self-harm or suicide, the AI should recognize this as beyond its capabilities and guide the patient to seek appropriate emergency care.
    10. By far the most important aspect of transcript generation is ensuring DEPTH and DEEP EXPLORATION and EXPLANATION. DO NOT interact on a SUPERFICIAL level.
    11. It is VERY IMPORTANT that the transcripts consist of 1:1 turns of the patient speaking, then the therapist. The therapist should not speak twice in a row without the patient having the opportunity to speak and vice versa. 
    12. Each individual part of the transcript DOES NOT REQUIRE TERMINATION or wrapping up UNLESS it is PART 4. Only wrap up the session with the patient at the end of PART 4.
    13. This part is a direct continuation of the previous parts. Maintain the flow of conversation and do not restart the session or introduce new major topics unless it's part 1.
    14. DO NOT include any phrases like 'here is part x' or anything similar.
    15. Avoid repetitive validation or reassurance statements. Ensure that empathy is conveyed in varied and non-repetitive ways.
    
    Specific instructions for this part of the assessment session:
    {get_assessment_part_instructions(session_number, part)}
    """

    for attempt in range(MAX_REGENERATION_ATTEMPTS):
        messages = [{"role": "user", "content": general_instructions}]

        response = rate_limited_api_call(messages, max_tokens=4096, temperature=0.4)
        transcript = response.choices[0].message.content

        if (not check_premature_wrap_up(transcript, part) and 
            not check_for_asterisk(transcript) and
            (part != 1 or check_for_agenda(transcript)) and
            (part != 4 or check_for_session_summary(transcript))):
            break

        if check_premature_wrap_up(transcript, part):
            logger.warning(f"Premature wrap-up detected in set {set_number}, session {session_number}, part {part}. Regenerating (attempt {attempt + 1}).")
            general_instructions += "\n\nWARNING: The previous generation attempted to wrap up the session prematurely. DO NOT conclude or summarize the session in this part. Continue the dialogue naturally without any endings or summaries."
        elif check_for_asterisk(transcript):
            logger.warning(f"Asterisk detected in set {set_number}, session {session_number}, part {part}. Regenerating (attempt {attempt + 1}).")
            general_instructions += "\n\nWARNING: The previous generation contained an asterisk (*) character. Do not use asterisks in your response."
        elif part == 1 and not check_for_agenda(transcript):
            logger.warning(f"Agenda not found in set {set_number}, session {session_number}, part {part}. Regenerating (attempt {attempt + 1}).")
            general_instructions += "\n\nWARNING: The previous generation did not include an agenda setting. Please ensure that an agenda or plan for the session is explicitly discussed in part 1."
        elif part == 4 and not check_for_session_summary(transcript):
            logger.warning(f"Session Summary not found in set {set_number}, session {session_number}, part {part}. Regenerating (attempt {attempt + 1}).")
            general_instructions += "\n\nWARNING: The previous generation did not include a Session Summary. Please ensure that a detailed Session Summary is provided at the end of part 4."

    if (check_premature_wrap_up(transcript, part) or 
        check_for_asterisk(transcript) or
        (part == 1 and not check_for_agenda(transcript)) or
        (part == 4 and not check_for_session_summary(transcript))):
        logger.error(f"Failed to generate a valid transcript after {MAX_REGENERATION_ATTEMPTS} attempts for set {set_number}, session {session_number}, part {part}.")

    word_count = len(transcript.split())
    logger.debug(f"Generated assessment transcript part {part} for set {set_number}, session {session_number} with {word_count} words")
    
    return {"transcript": transcript, "word_count": word_count}

def generate_cbt_session_part(
    set_number: int, 
    session_number: int, 
    patient_profile: str, 
    previous_summary: str, 
    part: int, 
    previous_parts: List[str],
    phase_techniques: List[str] = []
) -> Dict[str, str]:
    """
    Generates a part of a CBT session transcript for depression.

    Args:
    set_number (int): The current set number.
    session_number (int): The current session number.
    patient_profile (str): The patient's profile.
    previous_summary (str): Summary of the previous session.
    part (int): The current part of the session (1, 2, 3, or 4).
    previous_parts (List[str]): The content of the previous parts of the current session.
    phase_techniques (List[str]): List of CBT techniques for the current phase.

    Returns:
    Dict[str, str]: A dictionary containing the session transcript part and word count.
    """
    cbt_phase = get_cbt_phase(session_number)

    general_instructions = f"""Generate part {part} of 4 for a detailed and realistic transcript of CBT session {session_number} of {NUM_SESSIONS} for a patient with depression. 
    This is a {cbt_phase} phase session.
    Therapist name: {THERAPIST_NAME}
    Patient profile: {patient_profile}

    Previous session summary: {previous_summary}

    Previous parts of this session:
    {"".join(previous_parts)}

    CBT techniques for this phase:
    {', '.join(phase_techniques)}

    Important notes:
    1. This is part {part} of 4 of the current session. Ensure continuity with previous parts and DO NOT conclude the session unless this is part 4.
    2. Use the format "Patient: " for the patient's dialogue and "Therapist: " for the therapist's dialogue.
    3. This is a text-based interaction between an AI CBT therapist (Nova) and a patient. Do not include any non-verbal cues, pauses, or actions.
    4. The AI therapist should provide thorough explanations with examples when introducing new concepts.
    5. Regularly check the patient's understanding and explore any questions.
    6. Check in regularly to assess the patient's emotional state at various points in the session. 
    7. Incorporate the patient's specific background, goals, and challenges from their profile into the session content. Do not make the patient's improvement UNREALISTIC, specifically, do not RUSH the patient to wellness in your transcripts.
    8. The output should ONLY be the transcript dialogue, with no additional descriptions or explanations.
    9. If the patient expresses thoughts of self-harm or suicide, the AI should recognize this as beyond its capabilities and guide the patient to seek appropriate emergency care.
    10. Instead of providing actual resources or worksheets, the AI therapist should discuss homework and exercises in detail within the session.
    11. Slowly build on techniques and concepts introduced in previous sessions, showing progression throughout the therapy.
    12. By far the most important aspect of transcript generation is ensuring DEPTH and DEEP EXPLORATION and EXPLANATION. DO NOT interact on SUPERFICIAL level. This means that you should spend significant time exploring the patient's EMOTIONS, THOUGHTS and BEHAVIORS. DO NOT RUSH the sessions. AVOID BEING REPETITIVE, especially when providing validation or responding to the patient. Use varied language and approaches to maintain engagement.
    13. Emphasis should be placed on the connections between thoughts, feelings and behaviors (CBT triangle)
    14. It is VERY IMPORTANT that the transcripts consist of 1:1 turns of the patient speaking, then the therapist. The therapist should not speak twice in a row without the patient having the opportunity to speak and vice versa. 
    15. Each individual part of the transcript DOES NOT REQUIRE TERMINATION or wrapping up UNLESS it is PART 4. Only wrap up the session with the patient at the end of PART 4. It is okay to leave some themes unfinished in an individual part of the transcript, these can be completed in the next part. 
    16. This part is a direct continuation of the previous parts. Maintain the flow of conversation and do not restart the session or introduce new major topics unless it's part 1.
    17. When introducing a new technique, provide a thorough explanation, rationale, and examples. Ensure patient participation and understanding.
    18. From session 4 onwards, introduce and discuss cognitive distortions when relevant.
    19. In sessions 2 and 3, include discussion about the cognitive conceptualization (relationship between automatic thoughts, intermediate beliefs, core beliefs, and how they influence each other and how CBT aims to modify them).
    20. Reduce excessive reassuring statements and cycles of validation and gratitude. While empathy is important, focus on productive therapeutic work.
    21. DO NOT include phrases like "here is part x" in the transcript.

    {get_part_specific_instructions(part)}
    """

    for attempt in range(MAX_REGENERATION_ATTEMPTS):
        messages = [{"role": "user", "content": general_instructions}]

        response = rate_limited_api_call(messages, max_tokens=4096, temperature=0.4)
        transcript = response.choices[0].message.content

        if (not check_premature_wrap_up(transcript, part) and 
            not check_for_asterisk(transcript) and
            (part != 1 or check_for_agenda(transcript)) and
            (part != 4 or check_for_session_summary(transcript))):
            break

        if check_premature_wrap_up(transcript, part):
            logger.warning(f"Premature wrap-up detected in set {set_number}, session {session_number}, part {part}. Regenerating (attempt {attempt + 1}).")
            general_instructions += "\n\nWARNING: The previous generation attempted to wrap up the session prematurely. DO NOT conclude or summarize the session in this part. Continue the dialogue naturally without any endings or summaries."
        elif check_for_asterisk(transcript):
            logger.warning(f"Asterisk detected in set {set_number}, session {session_number}, part {part}. Regenerating (attempt {attempt + 1}).")
            general_instructions += "\n\nWARNING: The previous generation contained an asterisk (*) character. Do not use asterisks in your response."
        elif part == 1 and not check_for_agenda(transcript):
            logger.warning(f"Agenda not found in set {set_number}, session {session_number}, part {part}. Regenerating (attempt {attempt + 1}).")
            general_instructions += "\n\nWARNING: The previous generation did not include an agenda setting. Please ensure that an agenda or plan for the session is explicitly discussed in part 1."
        elif part == 4 and not check_for_session_summary(transcript):
            logger.warning(f"Session Summary not found in set {set_number}, session {session_number}, part {part}. Regenerating (attempt {attempt + 1}).")
            general_instructions += "\n\nWARNING: The previous generation did not include a Session Summary. Please ensure that a detailed Session Summary is provided at the end of part 4."

    if (check_premature_wrap_up(transcript, part) or 
        check_for_asterisk(transcript) or
        (part == 1 and not check_for_agenda(transcript)) or
        (part == 4 and not check_for_session_summary(transcript))):
        logger.error(f"Failed to generate a valid transcript after {MAX_REGENERATION_ATTEMPTS} attempts for set {set_number}, session {session_number}, part {part}.")

    word_count = len(transcript.split())
    logger.debug(f"Generated transcript part {part} for set {set_number}, session {session_number} with {word_count} words")

    return {"transcript": transcript, "word_count": word_count}

def generate_cbt_session(
    set_number: int, 
    session_number: int, 
    patient_profile: str, 
    previous_summary: str = ""
) -> Dict[str, str]:
    """
    Generates a full CBT session transcript for depression by combining four parts.

    Args:
    set_number (int): The current set number.
    session_number (int): The current session number.
    patient_profile (str): The patient's profile.
    previous_summary (str): Summary of the previous session, if any.

    Returns:
    Dict[str, str]: A dictionary containing the full session transcript and word count.
    """
    full_transcript = ""
    total_word_count = 0
    previous_parts = []

    current_phase = get_cbt_phase(session_number)
    phase_techniques = get_techniques_for_phase(current_phase)

    # Add the patient's first message with the verbatim summary for sessions 2 onwards
    if session_number > 1:
        first_message = f"Patient: Hi, I'm back for session {session_number}. Here is the summary you provided me at the end of our last session:\n\n{previous_summary}\n\n"
        full_transcript += first_message
        total_word_count += len(first_message.split())
        previous_parts.append(first_message)

    for part in range(1, 5):
        if session_number <= 3:
            part_data = generate_assessment_session_part(
                set_number, session_number, patient_profile, 
                previous_summary, part, previous_parts
            )
        else:
            part_data = generate_cbt_session_part(
                set_number, session_number, patient_profile, 
                previous_summary, part, previous_parts, phase_techniques
            )

        full_transcript += part_data["transcript"] + "\n\n"
        total_word_count += part_data["word_count"]
        previous_parts.append(part_data["transcript"])

        # Write part to file immediately
        set_dir = os.path.join(OUTPUT_DIR, f"depression_set_{set_number}")
        part_path = os.path.join(set_dir, f"session_{session_number}_part_{part}.json")
        with open(part_path, "w") as f:
            json.dump(part_data, f, indent=2)

    logger.info(f"Transcript completed: Set {set_number}, Session {session_number} - {total_word_count} words")

    return {
        "transcript": full_transcript, 
        "word_count": total_word_count
    }

def generate_cbt_set(set_number: int, progress_bar: tqdm):
    """
    Generates a full set of CBT sessions for a single patient with depression.

    Args:
    set_number (int): The current set number.
    progress_bar (tqdm): Progress bar object for updating progress.
    """
    set_dir = os.path.join(OUTPUT_DIR, f"depression_set_{set_number}")
    os.makedirs(set_dir, exist_ok=True)

    profile_path = os.path.join(set_dir, "patient_profile.json")
    if os.path.exists(profile_path):
        with open(profile_path, "r") as f:
            profile_data = json.load(f)
            patient_profile = profile_data["profile"]
            patient_characteristics = profile_data["characteristics"]
        logger.debug(f"Loaded existing patient profile for set {set_number}")
    else:
        patient_characteristics = generate_patient_characteristics()
        patient_profile = generate_patient_profile(patient_characteristics)
        with open(profile_path, "w") as f:
            json.dump({"profile": patient_profile, "characteristics": patient_characteristics}, f, indent=2)
        logger.debug(f"Generated new patient profile for set {set_number}")

    previous_summary = ""

    for session_num in range(1, NUM_SESSIONS + 1):
        session_path = os.path.join(set_dir, f"session_{session_num}.json")
        if os.path.exists(session_path):
            logger.debug(f"Skipping existing depression set {set_number}, session {session_num}")
            with open(session_path, "r") as f:
                session_data = json.load(f)
                previous_summary = extract_session_summary(session_data["transcript"])
        else:
            logger.debug(f"Generating depression set {set_number}, session {session_num}")
            session_data = generate_cbt_session(
                set_number, session_num, patient_profile, previous_summary
            )
            with open(session_path, "w") as f:
                json.dump(session_data, f, indent=2)
            previous_summary = extract_session_summary(session_data["transcript"])

        progress_bar.update(1)
        progress_bar.set_description(f"Set {set_number}, Session {session_num}")

    logger.info(f"Completed depression set {set_number}")


def main():
    """
    Main function to orchestrate the generation of all CBT session sets for depression.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_sessions = NUM_DEPRESSION_SETS * NUM_SESSIONS
    with tqdm(total=total_sessions, desc="Generating CBT Sessions", unit="session") as progress_bar:
        for i in range(NUM_DEPRESSION_SETS):
            generate_cbt_set(i + 1, progress_bar)

    logger.info("CBT session generation completed.")

if __name__ == "__main__":
    main()