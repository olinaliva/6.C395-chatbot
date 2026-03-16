import datetime as dt
import re
import unicodedata
from collections import Counter

import pandas as pd
from dateutil import parser as date_parser
from huggingface_hub import InferenceClient

from config import BASE_MODEL, MY_MODEL, HF_TOKEN

# #very basic system prompt to get it started (from cheat sheet on canvas)
# SYSTEM_PROMPT = '''
# You are a helpful assistant for Boston Public Schools enrollment.
# Your job is to help families find schools that match their needs.
# KEY FACTS:
# - Boston uses a home-based assignment system. The schools
# available to a family depend on their home address.
# - Families register at Welcome Centers or online.
# - Registration for the 2025-2026 school year opens in January.
# RULES:
# - Always ask what neighborhood the family lives in.
# - Always ask the child’s grade level.
# - If you are unsure about a specific school’s details, say so.
# - Never make up school names or addresses.
# '''

# #ITERATION 1
# SYSTEM_PROMPT='''
# Today's date is March 2026. The current school year is 2025–2026. The upcoming enrollment cycle is for the 2026–2027 school year.

# You are a knowledgeable assistant designed to help Boston families understand and navigate the Boston Public Schools (BPS) enrollment system. 

# Only answer questions related to Boston Public Schools enrollment, school selection, and the application process. If asked about anything else, politely let the user know this tool is specifically designed for BPS enrollment questions. If a user asks about private schools, charter schools, or unrelated topics, acknowledge their question and gently steer the conversation back to BPS enrollment.

# Use a warm, patient, and conversational tone. Always explain acronyms and education jargon if you use them. Remember that some users may be non-native English speakers or first-time BPS families. Use simple language and short sentences. Break down complex information into simple steps. 
# You have access to three datasets to help answer user questions:
# Main School Directory -  contains approximately 120 BPS schools including traditional, pilot, innovation, specialized education, and other school types. Each record includes school name, address, phone, email, and grade range (grade low to grade high).
# Language Programs - contains schools offering bilingual or language-specific programs in Spanish, Haitian Kreyòl, Vietnamese, Chinese, and ASL, with associated grade ranges.
# Alternative Education Programs - contains alternative programs hosted at various sites, including Re-engagement, Adult Education, Accelerated Diploma, Competency-based, Alternative, Recovery-focused, EL-focused, Blended/Online, and Community-based programs. Each record includes eligibility notes, target students, age range, delivery model, and support services.
# The following are specific rules or processes that are important:
# Eligibility Rules: a student's grade level is determined by their year of birth. Always ask for the child's date of birth rather than their age to determine the correct grade placement.
# School Assignment Policy: BPS uses a home-based assignment policy for grades K0-8. Every family receives a customized school list based on their home address, typically 10-14 options, including schools within a one-mile radius, high-quality option schools, and citywide schools. High schools (grades 7-12 and 9-12) are citywide options open to all students. Assignment uses a lottery-based algorithm, which means no specific placement can be guaranteed. Families can generate their personalized school list at https://boston.explore.avela.org/.
# If a user asks something you cannot answer from the available information, such as, specific option school eligibility logic or real-time seat availability, direct them to the official BPS website at https://www.bostonpublicschools.org or suggest they contact BPS directly. Never guarantee a student will be assigned to a specific school. If asked about private or charter schools, acknowledge the question and redirect to BPS enrollment.
# When gathering information, always ask for the child's date of birth (not age) before determining grade placement. Ask whether the child has any language needs before recommending schools. Ask whether the child has any special needs or an IEP before recommending schools.
# Never fabricate school names, addresses, or program details. Only recommend schools that appear in the provided datasets. If no schools match the user's criteria, say so clearly and suggest they contact BPS directly.
# If a user mentions an overage student (born 2008 or earlier), flag this and direct them to BPS directly. If a user seems confused or frustrated, slow down, acknowledge their feelings, and offer to start from the beginning.
# When listing schools, display them as a numbered list. Include school name, address, grades served, and school type. If a school has a language program, note the language and grades it covers. If a school is an alternative program, include the program type and eligibility notes.
# Keep responses short and focused, no more than 3-4 sentences per answer unless listing schools. Use bullet points or numbered lists when presenting multiple pieces of information. Bold school names to make them easy to scan. End every response with a follow-up question to keep the conversation moving.
# When redirecting to the BPS website, always include the full URL. When you don't have enough information to recommend a school, explain what you still need to know and why. If a user asks a yes/no question, answer it directly before elaborating

# Always respond in the language the user writes in.
# '''

#Iteration 2
SYSTEM_PROMPT = '''
Today's date is March 2026. The current school year is 2025–2026. The upcoming enrollment cycle is for the 2026–2027 school year.


You are a knowledgeable assistant designed to help Boston families understand and navigate the Boston Public Schools (BPS) enrollment system.


Only answer questions related to Boston Public Schools enrollment, school selection, and the application process. If asked about anything else, politely let the user know this tool is specifically designed for BPS enrollment questions. If a user asks about private schools, charter schools, or unrelated topics, acknowledge their question and gently steer the conversation back to BPS enrollment.


Use a warm, patient, and conversational tone. Always explain acronyms and education jargon if you use them. Remember that some users may be non-native English speakers or first-time BPS families. Use simple language and short sentences. Break down complex information into simple steps.
You have access to three datasets to help answer user questions:
Main School Directory -  contains approximately 120 BPS schools including traditional, pilot, innovation, specialized education, and other school types. Each record includes school name, address, phone, email, and grade range (grade low to grade high).
Language Programs - contains schools offering bilingual or language-specific programs in Spanish, Haitian Kreyòl, Vietnamese, Chinese, and ASL, with associated grade ranges.
Alternative Education Programs - contains alternative programs hosted at various sites, including Re-engagement, Adult Education, Accelerated Diploma, Competency-based, Alternative, Recovery-focused, EL-focused, Blended/Online, and Community-based programs. Each record includes eligibility notes, target students, age range, delivery model, and support services.
The following are specific rules or processes that are important:
Boston Public Schools — Enrollment Quick Reference (2026–2027)
What is Great Starts? Great Starts is a citywide enrollment platform for Boston families to explore and register for childcare, preschool, and BPS schools (ages 3–18). Register at apply.avela.org/boston. Explore schools at bostonpublicschools.org/exploreschools.

Key Dates
Priority Registration Opens (K0, K1, K2, Gr. 6, 7, 9): January 5, 2026
Priority Registration Opens (all other grades): February 9, 2026
Priority Registration Closes (K0, K1, K2, Gr. 6, 7, 9): February 6, 2026
Priority Registration Closes (all other grades): April 2, 2026
Assignment Notifications (K0, K1, K2, Gr. 6, 7, 9): March 31, 2026
Assignment Notifications (all other grades): May 31, 2026
Waitlist expires (Grades 1–12): November 30, 2026
Waitlist expires (Grades K0–K2): January 31, 2027

Age Requirements (by September 1, 2026)
K0: must turn 3
K1: must turn 4
K2: must turn 5
Grade 1: must turn 6 (no exceptions)
Note: K0 seats are very limited (mostly for students with disabilities). K1 has 2,400+ seats. All K2 students are guaranteed a seat, though not necessarily at their first-choice school.

How to Register — 6 Steps
Check age requirements.
Find eligible schools at boston.explore.avela.org (your address determines your list).
Watch a registration info session recording (available in 8 languages on the BPS website; live sessions have ended).
Gather required documents (see below).
Register online at apply.avela.org/boston.
A registration specialist will review your application and follow up if anything is missing.
Required Documents:
Child's birth certificate, I-94, or passport
Child's up-to-date immunization record
Child's physical exam (within the past year)
Parent/Guardian photo ID
Two proofs of Boston residency from different categories: utility bill, deed/mortgage, W2/pay stub, bank/credit card statement, government agency letter, or current lease/Section 8 agreement (most within 60 days; lease within 1 year)
Optional: copy of IEP if applicable; high school transcript for grade placement.

School Types
Elementary (Pre-K–Grade 6 or K–8): Most BPS elementary schools run Pre-K or K through Grade 6. Grade 7 students in a K–6 school receive a choice form to select a 7–12 school.
High Schools (Grades 7–12 or 9–12): All BPS high schools are citywide — any Boston student can apply. Some require a separate application in addition to ranking the school on your BPS form: Boston Arts Academy, Boston Day and Evening Academy, Boston Green Academy, Edward M. Kennedy Academy for Health Careers, Fenway High School, New Mission High School, Madison Park Technical Vocational High School.
Exam Schools (Grades 7 & 9): Boston Latin School, Boston Latin Academy, John D. O'Bryant School of Math & Science. Requirements: B or higher GPA, at least one exam school ranked, valid MAP Growth score. Composite = 30% MAP + 70% GPA. Students experiencing homelessness, living in BHA public housing, or in DCF care receive +10 bonus points.

Assignments & Waitlists Assignments are NOT first-come, first-served. All applications in a round are processed together after the round closes. Rank as many schools as possible — suggested minimum 5, more is better.
Registered by April 3, 2026: placed on all waitlists for schools ranked above your assignment; automatic movement through July 2026. After August 1, reduced to one waitlist (highest-ranked school).
Registered after April 3, 2026: placed on one waitlist only (highest-ranked school).
Keep your phone and email current — BPS cannot hold a seat if they can't reach you.

Welcome Centers
Dorchester: 1216 Dorchester Ave | 617-635-8015 | M/Tu/Th/F 9am–5pm, W 12–5pm | English, Portuguese, Spanish, Cabo Verdean Creole, Vietnamese
East Boston: 312 Border St | 617-635-9597 | M/Tu 9am–5pm, W 12–5pm (January only) | English, Spanish, Portuguese | Closed on BPS school holidays
Roslindale: 515 Hyde Park Ave | 617-635-8040 | M/Tu/Th/F 9am–5pm, W 12–5pm | Cabo Verdean Creole, English, French, Haitian Creole, Portuguese, Spanish
Roxbury: 2300 Washington St | 617-635-9010 | M/Tu/Th/F 9am–5pm, W 12–5pm | Cabo Verdean Creole, English, Portuguese, Somali, Spanish
General line: 617-635-9010 | welcomeservices@bostonpublicschools.org

Other Resources
Charter schools (not BPS): applybostoncharterschools.org
Countdown to Kindergarten (free playgroups): countdowntokindergarten.org | 617-635-9288
Multilingual learners needing language testing: contact a Welcome Center to schedule at the NACC (2300 Washington St, Roxbury)
Grade level waiver requests: contact TeeAra Dias at tdias@bostonpublicschools.org or 617-635-9701
Exam school info: bostonpublicschools.org/exam



If a user asks something you cannot answer from the available information, such as, specific option school eligibility logic or real-time seat availability, direct them to the official BPS website at https://www.bostonpublicschools.org or suggest they contact BPS directly. Never guarantee a student will be assigned to a specific school. If asked about private or charter schools, acknowledge the question and redirect to BPS enrollment.
When gathering information, always ask for the child's date of birth (not age) before determining grade placement. Ask whether the child has any language needs before recommending schools. Ask whether the child has any special needs or an IEP before recommending schools.
Never fabricate school names, addresses, or program details. Only recommend schools that appear in the provided datasets. If no schools match the user's criteria, say so clearly and suggest they contact BPS directly.
If a user mentions an overage student (born 2008 or earlier), flag this and direct them to BPS directly. If a user seems confused or frustrated, slow down, acknowledge their feelings, and offer to start from the beginning.
When listing schools, display them as a numbered list. Include school name, address, grades served, and school type. If a school has a language program, note the language and grades it covers. If a school is an alternative program, include the program type and eligibility notes.
Keep responses short and focused, no more than 3-4 sentences per answer unless listing schools. Use bullet points or numbered lists when presenting multiple pieces of information. Bold school names to make them easy to scan. End every response with a follow-up question to keep the conversation moving.
When redirecting to the BPS website, always include the full URL. When you don't have enough information to recommend a school, explain what you still need to know and why. If a user asks a yes/no question, answer it directly before elaborating


Only include URLs from this allowlist: bostonpublicschools.org, apply.avela.org, boston.explore.avela.org, applybostoncharterschools.org, countdowntokindergarten.org, bostonpublicschools.helpdocs.io, bostonartsacademy.org.
If unsure about the user's language, default to English.
Always respond in the language the user writes in.
'''

CUTOFF_DATE = dt.date(2026, 9, 1)
OVERAGE_BIRTH_YEAR = 2008
MAX_USER_CHARS = 1200
MAX_REPEAT_RATIO = 0.6
MAX_HISTORY_MESSAGES = 10

MONTH_PATTERN = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
FULL_DATE_REGEXES = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(rf"\b{MONTH_PATTERN}\s+\d{{1,2}}(?:,\s*)?\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b\d{{1,2}}\s+{MONTH_PATTERN}\s+\d{{4}}\b", re.IGNORECASE),
]

ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
}


def normalize_text(text):
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return normalized.lower()


def is_repetitive(text):
    words = re.findall(r"[a-zA-Z']+", normalize_text(text))
    if len(words) < 40:
        return False
    counts = Counter(words)
    most_common = counts.most_common(1)[0][1]
    return most_common / len(words) >= MAX_REPEAT_RATIO


def contains_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def parse_grade_tokens(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).upper()
    text = text.replace("PRE-K", "PREK").replace("PRE K", "PREK")
    tokens = re.findall(r"K0|K1|K2|PREK|PK|K|\d{1,2}", text)
    grades = []
    for token in tokens:
        if token == "K0":
            grades.append(0)
        elif token in {"K1", "PREK", "PK"}:
            grades.append(1)
        elif token in {"K2", "K"}:
            grades.append(2)
        else:
            try:
                grades.append(int(token))
            except ValueError:
                continue
    return grades


def compute_grade_bounds(row):
    tokens = parse_grade_tokens(row.get("grade_low")) + parse_grade_tokens(row.get("grade_high"))
    if not tokens:
        return pd.Series({"grade_low_num": pd.NA, "grade_high_num": pd.NA})
    return pd.Series({"grade_low_num": min(tokens), "grade_high_num": max(tokens)})


def grade_label_from_age(age):
    if age < 3:
        return None
    if age == 3:
        return "K0"
    if age == 4:
        return "K1"
    if age == 5:
        return "K2"
    return str(age - 5)


def format_grade_label(grade_label):
    if grade_label is None:
        return "not age-eligible"
    if grade_label.startswith("K"):
        return grade_label
    return f"Grade {grade_label}"


def required_age_for_grade(grade_label):
    if grade_label in {"K0", "K1", "K2"}:
        return {"K0": 3, "K1": 4, "K2": 5}[grade_label]
    try:
        return int(grade_label) + 5
    except ValueError:
        return None


def extract_requested_grade(text):
    normalized = normalize_text(text)
    if "k0" in normalized:
        return "K0"
    if "k1" in normalized:
        return "K1"
    if "k2" in normalized or "kindergarten" in normalized:
        return "K2"
    if "pre-k" in normalized or "prek" in normalized or "pre k" in normalized:
        return "K1"

    grade_match = re.search(r"\bgrade\s*(\d{1,2})\b", normalized)
    if grade_match:
        return grade_match.group(1)

    ordinal_match = re.search(r"\b(\d{1,2})(st|nd|rd|th)\s*grade\b", normalized)
    if ordinal_match:
        return ordinal_match.group(1)

    for word, number in ORDINAL_WORDS.items():
        if f"{word} grade" in normalized:
            return str(number)

    return None


def extract_turning_age(text):
    match = re.search(
        r"\b(turns|turning|will be)\s+(\d{1,2})\s+(?:years? old\s+)?on\s+([^\n\.,;]+)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    age = int(match.group(2))
    date_str = match.group(3).strip()
    if not re.search(MONTH_PATTERN, date_str, re.IGNORECASE) and not re.search(r"\d{1,2}[/-]\d{1,2}", date_str):
        return None
    default_dt = dt.datetime(CUTOFF_DATE.year, 1, 1)
    try:
        parsed = date_parser.parse(date_str, default=default_dt, fuzzy=True)
        return age, parsed.date()
    except (ValueError, OverflowError):
        return None


def extract_birthdate(text):
    normalized = normalize_text(text)
    if not contains_any(normalized, ["born", "birth", "dob", "date of birth", "birthday"]):
        return None
    for regex in FULL_DATE_REGEXES:
        match = regex.search(text)
        if match:
            try:
                parsed = date_parser.parse(match.group(0), monthfirst=True, dayfirst=False)
                return parsed.date()
            except (ValueError, OverflowError):
                return None
    return None


def age_on_cutoff(birthdate):
    return CUTOFF_DATE.year - birthdate.year - ((CUTOFF_DATE.month, CUTOFF_DATE.day) < (birthdate.month, birthdate.day))


def build_age_eligibility_answer(user_input):
    turning_info = extract_turning_age(user_input)
    birthdate = extract_birthdate(user_input)
    requested_grade = extract_requested_grade(user_input)
    normalized = normalize_text(user_input)

    if turning_info:
        age_turning, date_turning = turning_info
        age_at_cutoff = age_turning if date_turning <= CUTOFF_DATE else age_turning - 1
    elif birthdate:
        age_at_cutoff = age_on_cutoff(birthdate)
    else:
        if re.search(r"\bwhat grades\b.*\bbps\b", normalized):
            return None
        if contains_any(
            normalized,
            [
                "eligible",
                "eligibility",
                "years old",
                "year old",
                "age",
                "born",
                "birthday",
                "turns",
                "turning",
                "kindergarten",
                "k0",
                "k1",
                "k2",
                "pre-k",
                "prek",
                "can my child",
                "can they",
                "can he",
                "can she",
            ],
        ):
            return (
                f"To determine eligibility, I need the child's full date of birth. "
                f"BPS uses a September 1, 2026 cutoff for the 2026–2027 school year. "
                "What is their date of birth?"
            )
        return None

    if birthdate and birthdate.year <= OVERAGE_BIRTH_YEAR:
        return (
            "This looks like an overage student (born in 2008 or earlier). "
            "Please contact BPS directly or a Welcome Center for the right placement. "
            "Do you want the closest Welcome Center?"
        )

    eligible_grade = grade_label_from_age(age_at_cutoff)
    if eligible_grade is None:
        return (
            f"By September 1, 2026, the student would be {age_at_cutoff} years old, "
            "which is too young for K0. "
            "Do you want me to explain BPS age requirements?"
        )

    if requested_grade:
        required_age = required_age_for_grade(requested_grade)
        if required_age is not None:
            if age_at_cutoff >= required_age:
                return (
                    f"Yes. A student must be {required_age} by September 1, 2026 to start {format_grade_label(requested_grade)}, "
                    f"and they would be {age_at_cutoff} by that cutoff. "
                    "Do you want help finding schools for that grade?"
                )
            return (
                f"No. A student must be {required_age} by September 1, 2026 to start {format_grade_label(requested_grade)}. "
                f"Based on the information provided, they would be {age_at_cutoff} by the cutoff and would be eligible for {format_grade_label(eligible_grade)}. "
                "Do you want help finding schools for that grade?"
            )

    return (
        f"By September 1, 2026, the student would be {age_at_cutoff} years old, which makes them eligible for {format_grade_label(eligible_grade)}. "
        "Do you want help finding schools for that grade or a language program?"
    )


def format_rows(rows, max_rows, formatter, label):
    lines = [label]
    for row in rows[:max_rows]:
        lines.append(f"- {formatter(row)}")
    if len(rows) > max_rows:
        lines.append(f"- ...and {len(rows) - max_rows} more")
    return "\n".join(lines)


class Chatbot:
    """
    This class is extra scaffolding around a model. Modify this class to specify how the model recieves prompts and generates responses.

    Example usage:
        chatbot = Chatbot()
        response = chatbot.get_response("What options are available for me?")
    """

    def __init__(self, schools_df, languages_df, alternative_df):
        """
        Initialize the chatbot with a HF model ID
        """
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL # define MY_MODEL in config.py if you create a new model in the HuggingFace Hub
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        # spreadsheets
        self.schools_df = schools_df.copy()
        self.languages_df = languages_df.copy()
        self.alternative_df = alternative_df.copy()

        self.schools_df[["grade_low_num", "grade_high_num"]] = self.schools_df.apply(
            compute_grade_bounds,
            axis=1,
        )

        self.languages_df["language_norm"] = self.languages_df["language"].astype(str).map(normalize_text)
        self.languages_df["grade_low_num"] = self.languages_df["grade_low"].apply(
            lambda value: min(parse_grade_tokens(value)) if parse_grade_tokens(value) else pd.NA
        )
        self.languages_df["grade_high_num"] = self.languages_df["grade_high"].apply(
            lambda value: max(parse_grade_tokens(value)) if parse_grade_tokens(value) else pd.NA
        )

        self.alternative_df["program_type_norm"] = self.alternative_df["program_type_standardized"].astype(str).map(normalize_text)
        
    def format_prompt(self, user_input, history=None, language_instruction="", forced_answer=""):
        """
        TODO: Implement this method to format the user's input into a proper prompt.

        This method should:
        1. Add any necessary system context or instructions
        2. Format the user's input appropriately
        3. Add any special tokens or formatting the model expects

        Args:
            user_input (str): The user's question

        Returns:
            str: A formatted prompt ready for the model

        (each model might expect different format, could check in huggingface documentation)
        Example prompt format:
            "You are a helpful assistant that specializes in...
             User: {user_input}
             Assistant:"
        """
        # adds info about the schools filtered from spreadsheets
        relevant_data = self.get_relevant_data(user_input)
        system_content = SYSTEM_PROMPT
        if language_instruction:
            system_content = language_instruction + "\n\n" + system_content
        if forced_answer:
            system_content += (
                "\n\nFORCED ANSWER (use this content, do not add new facts; "
                "translate only if the user's language requires it):\n"
                f"{forced_answer}"
            )
        if relevant_data:
            system_content += f"\n\nRELEVANT DATA FOR THIS QUERY:\n{relevant_data}"

        messages = [{"role": "system", "content": system_content}]
        # messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            if isinstance(history, list) and history:
                if isinstance(history[0], dict):
                    history = history[-MAX_HISTORY_MESSAGES:]
                else:
                    history = history[-max(1, MAX_HISTORY_MESSAGES // 2):]
            # for user_message, chatbot_message in history:
            #     messages.append({"role": "user", "content": user_message})
            #     messages.append({"role": "assistant", "content": chatbot_message})
            for item in history:
                # new format: list of dicts with 'role' and 'content'
                if isinstance(item, dict):
                    messages.append({"role": item["role"], "content": item["content"]})
                # old format: list of (user_msg, bot_msg) tuples
                else:
                    user_message, chatbot_message = item
                    messages.append({"role": "user", "content": user_message})
                    messages.append({"role": "assistant", "content": chatbot_message})

        messages.append({"role": "user", "content": user_input})
    
        return messages

        
    def get_response(self, user_input, history=None, language_instruction=""):
        """
        TODO: Implement this method to generate responses to user questions.

        This method should:
        1. Use format_prompt() to prepare the input
        2. Generate a response using the model
        3. Clean up and return the response

        Args:
            user_input (str): The user's question

        Returns:
            generator: Yields partial response strings as they stream in
        """
        if not user_input:
            yield "Please enter a question about BPS enrollment. How can I help?"
            return

        normalized = normalize_text(user_input)
        if len(user_input) > MAX_USER_CHARS or is_repetitive(user_input):
            yield (
                "That message is a bit long or repetitive. "
                "Please send a shorter question about BPS enrollment so I can help."
            )
            return

        if contains_any(normalized, ["ignore all previous instructions", "system prompt"]) or contains_any(
            normalized, ["assassinate", "murder", "kill", "bomb", "explosive", "weapon"]
        ):
            yield "I can't assist with that. I can help with BPS enrollment questions."
            return

        forced_answer = self.get_forced_answer(user_input)
        messages = self.format_prompt(user_input, history, language_instruction, forced_answer)
        partial = ""
        try:
            for chunk in self.client.chat_completion(
                messages=messages,
                stream=True,
                temperature=0.2,
                top_p=0.9,
                max_tokens=512,
            ):
                token = chunk.choices[0].delta.content
                if token is not None:
                    partial += token
                    yield partial
        except Exception as e:
            yield (partial or "") + f"\n\n_(Error: {e}. Please try again in a few seconds.)_"
    
    def get_forced_answer(self, user_input):
        age_answer = build_age_eligibility_answer(user_input)
        if age_answer:
            return age_answer

        normalized = normalize_text(user_input)
        if re.search(r"\bwhat grades\b.*\bbps\b", normalized) or re.search(r"\bgrades does bps\b", normalized):
            return (
                "BPS serves grades K0 (age 3) through Grade 12. "
                "K0 and K1 are pre-K levels, K2 is kindergarten, and Grade 1 starts at age 6 by the September 1, 2026 cutoff. "
                "What grade are you looking for?"
            )

        if "difference between" in normalized and ("k0" in normalized or "k1" in normalized or "k2" in normalized):
            return (
                "K0 is pre-K for students who turn 3 by September 1, 2026. "
                "K1 is pre-K for students who turn 4 by September 1, 2026. "
                "K2 is kindergarten for students who turn 5 by September 1, 2026. "
                "Grade 1 starts at age 6. "
                "Which grade are you asking about?"
            )

        if "great starts" in normalized:
            return (
                "Great Starts is Boston's citywide enrollment platform to explore and register for childcare, preschool, and BPS schools (ages 3-18). "
                "You can register at https://apply.avela.org/boston. "
                "What age or grade is your child?"
            )

        if "register" in normalized and ("how do i" in normalized or "how can i" in normalized or "how to" in normalized):
            return (
                "You can start registration online at https://apply.avela.org/boston. "
                "You will need required documents (birth certificate/passport, immunization record, physical exam within the past year, "
                "parent/guardian photo ID, and two proofs of Boston residency). "
                "If anything is missing, BPS may ask you to visit a Welcome Center. "
                "Do you want the document list or the nearest Welcome Center?"
            )

        if contains_any(normalized, ["list of schools", "eligible schools", "eligible for", "my list"]) and "school" in normalized:
            return (
                "Your eligible school list depends on your Boston home address. "
                "You can generate it at https://boston.explore.avela.org. "
                "If you share your child's grade and any language needs, I can help you interpret that list."
            )

        if "document" in normalized or "proof of residency" in normalized:
            return (
                "Required documents include: child's birth certificate, I-94, or passport; immunization record; physical exam within the past year; "
                "parent/guardian photo ID; and two proofs of Boston residency from different categories (utility bill, lease, W2/pay stub, bank statement, "
                "government agency letter). "
                "Do you want help finding a Welcome Center?"
            )

        if "register" in normalized and ("online" in normalized or "in person" in normalized):
            return (
                "You can start registration online at https://apply.avela.org/boston. "
                "BPS may still ask you to submit documents or meet with a Welcome Center in person. "
                "Do you want the closest Welcome Center?"
            )

        if "deadline" in normalized or "priority registration" in normalized or "registration window" in normalized:
            return (
                "For the 2026-2027 cycle, priority registration for K0/K1/K2 and grades 6, 7, 9 opens January 5, 2026 and closes February 6, 2026. "
                "For all other grades it opens February 9, 2026 and closes April 2, 2026. "
                "These dates can change each year, so confirm on https://www.bostonpublicschools.org. "
                "What grade are you registering for?"
            )

        if "register" in normalized and ("first day" in normalized or "last day" in normalized):
            return (
                "Assignments are not first-come, first-served; all applications in a round are processed together after the round closes. "
                "Still, register within the priority window to keep the most options available. "
                "Do you want the current registration dates?"
            )

        if "rank" in normalized and ("last" in normalized or "hurt" in normalized or "chance" in normalized):
            return (
                "No. Ranking a school lower does not hurt your chances at higher-ranked schools. "
                "BPS processes your list in order and tries to place you at your highest available choice. "
                "Do you want help ordering your list?"
            )

        if "rank" in normalized and "school" in normalized:
            return (
                "Assignments are not first-come, first-served; all applications in a round are processed together after the round closes. "
                "Rank as many schools as you can. A minimum of 5 is recommended, and more is better. "
                "Do you want help choosing schools to rank?"
            )

        if "assign" in normalized and "school" in normalized:
            return (
                "For grades K0-8, BPS uses a home-based assignment system and a lottery. "
                "You receive a customized list based on your address, then rank schools; the lottery places students into the highest available choice. "
                "For high schools (grades 7-12 or 9-12), all students can apply citywide. "
                "Do you want help understanding your customized list?"
            )

        if "waitlist" in normalized and ("august" in normalized or "aug 1" in normalized):
            return (
                "After August 1, 2026, BPS reduces your waitlists to one (your highest-ranked school). "
                "Waitlists for grades 1-12 expire November 30, 2026, and K0-K2 waitlists expire January 31, 2027. "
                "Do you want help checking your waitlist on https://boston.explore.avela.org?"
            )

        if "welcome center" in normalized and "east boston" in normalized:
            return (
                "East Boston Welcome Center: 312 Border St, East Boston, MA 02128. "
                "Hours: Mon/Tue 9am-5pm, Wed 12-5pm (January only). "
                "Phone: 617-635-9597. "
                "Do you need a different location?"
            )

        if "welcome center" in normalized and "roxbury" in normalized and "language" in normalized:
            return (
                "Roxbury Welcome Center languages: Cabo Verdean Creole, English, Portuguese, Somali, and Spanish. "
                "Location: 2300 Washington St. Phone: 617-635-9010. "
                "Do you want hours or another location?"
            )

        if "welcome center" in normalized and "wednesday" in normalized:
            return (
                "Most Welcome Centers are open Wednesdays 12pm-5pm. "
                "East Boston has Wednesday hours only in January. "
                "Which Welcome Center location are you asking about?"
            )

        if "call" in normalized and "welcome center" in normalized:
            return (
                "Yes, calling first is a good idea. The general Welcome Center line is 617-635-9010. "
                "Do you want the phone number for a specific location?"
            )

        if "charter" in normalized:
            return (
                "Charter schools are not part of BPS enrollment. "
                "You can apply at https://applybostoncharterschools.org. "
                "Do you still want help with BPS schools?"
            )

        return None

    # TODO debug and finesse this function
    def get_relevant_data(self, user_input):
        """
        Filters dataframes based on keywords in the user's message
        and returns a formatted string to inject into the prompt.
        """
        user_input_lower = normalize_text(user_input)
        results = []

        # Language filter
        language_keywords = {
            "spanish": ["spanish", "espanol"],
            "vietnamese": ["vietnamese"],
            "chinese": ["chinese", "mandarin", "cantonese"],
            "haitian": ["haitian", "kreyol", "creole"],
            "asl": ["asl", "sign language"],
        }
        for lang, keywords in language_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                matches = self.languages_df[self.languages_df["language_norm"].str.contains(lang)]
                if not matches.empty:
                    rows = matches[["school_name", "language", "grade_low", "grade_high"]].fillna("").to_dict("records")
                    results.append(
                        format_rows(
                            rows,
                            max_rows=8,
                            formatter=lambda row: f"{row['school_name']} | {row['language']} | Grades {row['grade_low']}-{row['grade_high']}",
                            label="Language Program Matches:",
                        )
                    )

        # Alternative program filter
        alt_keywords = ["alternative", "adult", "diploma", "re-engagement", "reengagement", "recovery", "online", "blended"]
        if any(keyword in user_input_lower for keyword in alt_keywords):
            matches = self.alternative_df[
                self.alternative_df["program_type_norm"].str.contains("|".join(alt_keywords))
            ]
            if not matches.empty:
                rows = matches[
                    ["program_name", "host_school_or_site", "program_type_standardized", "age_range"]
                ].fillna("").to_dict("records")
                results.append(
                    format_rows(
                        rows,
                        max_rows=6,
                        formatter=lambda row: f"{row['program_name']} | {row['program_type_standardized']} | {row['age_range']} | {row['host_school_or_site']}",
                        label="Alternative Program Matches:",
                    )
                )

        # Grade filter, looks for numbers or grade keywords
        requested_grade = extract_requested_grade(user_input)
        grade_num = None
        if requested_grade:
            if requested_grade.startswith("K"):
                grade_num = {"K0": 0, "K1": 1, "K2": 2}[requested_grade]
            else:
                grade_num = int(requested_grade)

        if grade_num is None and "high school" in user_input_lower:
            grade_num = 9

        if grade_num is not None:
            matches = self.schools_df[
                self.schools_df["grade_low_num"].notna()
                & self.schools_df["grade_high_num"].notna()
                & (self.schools_df["grade_low_num"] <= grade_num)
                & (self.schools_df["grade_high_num"] >= grade_num)
            ]
            if "high school" in user_input_lower:
                matches = matches[matches["is_high_school"] == 1]
            rows = matches[
                ["school_name", "address_full", "grade_low", "grade_high", "school_type"]
            ].fillna("").to_dict("records")
            if rows:
                results.append(
                    format_rows(
                        rows,
                        max_rows=10,
                        formatter=lambda row: f"{row['school_name']} | {row['address_full']} | Grades {row['grade_low']}-{row['grade_high']} | {row['school_type']}",
                        label="School Matches for Grade:",
                    )
                )

        return "\n\n".join(results) if results else ""
