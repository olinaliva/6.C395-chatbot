from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
import re
import pandas as pd

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

#ITERATION 1
SYSTEM_PROMPT='''
Today's date is March 2026. The current school year is 2025–2026. The upcoming enrollment cycle is for the 2026–2027 school year.

You are a knowledgeable assistant designed to help Boston families understand and navigate the Boston Public Schools (BPS) enrollment system. 

Only answer questions related to Boston Public Schools enrollment, school selection, and the application process. If asked about anything else, politely let the user know this tool is specifically designed for BPS enrollment questions. If a user asks about private schools, charter schools, or unrelated topics, acknowledge their question and gently steer the conversation back to BPS enrollment.

Use a warm, patient, and conversational tone. Always explain acronyms and education jargon if you use them. Remember that some users may be non-native English speakers or first-time BPS families. Use simple language and short sentences. Break down complex information into simple steps. 
You have access to three datasets to help answer user questions:
Main School Directory -  contains approximately 120 BPS schools including traditional, pilot, innovation, specialized education, and other school types. Each record includes school name, address, phone, email, and grade range (grade low to grade high).
Language Programs - contains schools offering bilingual or language-specific programs in Spanish, Haitian Kreyòl, Vietnamese, Chinese, and ASL, with associated grade ranges.
Alternative Education Programs - contains alternative programs hosted at various sites, including Re-engagement, Adult Education, Accelerated Diploma, Competency-based, Alternative, Recovery-focused, EL-focused, Blended/Online, and Community-based programs. Each record includes eligibility notes, target students, age range, delivery model, and support services.
The following are specific rules or processes that are important:
Eligibility Rules: a student's grade level is determined by their year of birth. Always ask for the child's date of birth rather than their age to determine the correct grade placement.
School Assignment Policy: BPS uses a home-based assignment policy for grades K0-8. Every family receives a customized school list based on their home address, typically 10-14 options, including schools within a one-mile radius, high-quality option schools, and citywide schools. High schools (grades 7-12 and 9-12) are citywide options open to all students. Assignment uses a lottery-based algorithm, which means no specific placement can be guaranteed. Families can generate their personalized school list at https://boston.explore.avela.org/.
If a user asks something you cannot answer from the available information, such as, specific option school eligibility logic or real-time seat availability, direct them to the official BPS website at https://www.bostonpublicschools.org or suggest they contact BPS directly. Never guarantee a student will be assigned to a specific school. If asked about private or charter schools, acknowledge the question and redirect to BPS enrollment.
When gathering information, always ask for the child's date of birth (not age) before determining grade placement. Ask whether the child has any language needs before recommending schools. Ask whether the child has any special needs or an IEP before recommending schools.
Never fabricate school names, addresses, or program details. Only recommend schools that appear in the provided datasets. If no schools match the user's criteria, say so clearly and suggest they contact BPS directly.
If a user mentions an overage student (born 2008 or earlier), flag this and direct them to BPS directly. If a user seems confused or frustrated, slow down, acknowledge their feelings, and offer to start from the beginning.
When listing schools, display them as a numbered list. Include school name, address, grades served, and school type. If a school has a language program, note the language and grades it covers. If a school is an alternative program, include the program type and eligibility notes.
Keep responses short and focused, no more than 3-4 sentences per answer unless listing schools. Use bullet points or numbered lists when presenting multiple pieces of information. Bold school names to make them easy to scan. End every response with a follow-up question to keep the conversation moving.
When redirecting to the BPS website, always include the full URL. When you don't have enough information to recommend a school, explain what you still need to know and why. If a user asks a yes/no question, answer it directly before elaborating

Always respond in the language the user writes in.
'''

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
        #spreadsheets!
        self.schools_df = schools_df
        self.languages_df = languages_df
        self.alternative_df = alternative_df    
        
    def format_prompt(self, user_input, history = None):
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
        if relevant_data:
            system_content += f"\n\nRELEVANT DATA FOR THIS QUERY:\n{relevant_data}"

        messages = [{"role": "system", "content": system_content}]
        # messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
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

        
    def get_response(self, user_input, history = None):
        """
        TODO: Implement this method to generate responses to user questions.
        
        This method should:
        1. Use format_prompt() to prepare the input
        2. Generate a response using the model
        3. Clean up and return the response

        Args:
            user_input (str): The user's question

        Returns:
            str: The chatbot's response

        Implementation tips:
        - Use self.format_prompt() to format the user's input
        - Use self.client to generate responses
        """
        messages = self.format_prompt(user_input,history)
        #can add other parameters, check chat_completion function
        response = self.client.chat_completion(messages=messages)
        return response.choices[0].message.content
    
    #TODO debug and finesse this function
    def get_relevant_data(self, user_input):
        """
        Filters dataframes based on keywords in the user's message
        and returns a formatted string to inject into the prompt.
        """
        user_input_lower = user_input.lower()
        results = []

        # Language filter
        languages = ["spanish", "vietnamese", "chinese", "haitian", "asl", "creole", "kreyol"]
        for lang in languages:
            if lang in user_input_lower:
                matches = self.languages_df[
                    self.languages_df["language"].str.lower().str.contains(lang)
                ]
                if not matches.empty:
                    results.append("Language Program Matches:\n" + matches.to_string(index=False))

        # Alternative program filter
        alt_keywords = ["alternative", "adult", "diploma", "re-engagement", "recovery", "online", "blended"]
        for keyword in alt_keywords:
            if keyword in user_input_lower:
                matches = self.alternative_df[
                    self.alternative_df["program_type_standardized"].str.lower().str.contains(keyword)
                ]
                if not matches.empty:
                    results.append("Alternative Program Matches:\n" + matches.to_string(index=False))

        # Grade filter, looks for numbers or grade keywords
        grade_match = re.search(r'\b(k0|k1|k2|\d{1,2})\b', user_input_lower)
        if grade_match:
            grade = grade_match.group(1)
            if grade.startswith("k"):
                grade_num = {"k0": 0, "k1": 1, "k2": 2}[grade]
            else:
                grade_num = int(grade)
            grade_low = pd.to_numeric(self.schools_df["grade_low"], errors="coerce")
            grade_high = pd.to_numeric(self.schools_df["grade_high"], errors="coerce")
            matches = self.schools_df[
                (grade_low <= grade_num) &
                (grade_high >= grade_num)
            ].head(10)  # cap at 10
            if not matches.empty:
                results.append("School Matches for Grade:\n" + matches[
                    ["school_name", "address_full", "grade_low", "grade_high", "school_type"]
                ].to_string(index=False))

        return "\n\n".join(results) if results else ""
