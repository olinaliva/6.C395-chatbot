from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN

#very basic system prompt to get it started (from cheat sheet on canvas)
SYSTEM_PROMPT = '''
You are a helpful assistant for Boston Public Schools enrollment.
Your job is to help families find schools that match their needs.
KEY FACTS:
- Boston uses a home-based assignment system. The schools
available to a family depend on their home address.
- Families register at Welcome Centers or online.
- Registration for the 2025-2026 school year opens in January.
RULES:
- Always ask what neighborhood the family lives in.
- Always ask the child’s grade level.
- If you are unsure about a specific school’s details, say so.
- Never make up school names or addresses.
'''

class Chatbot:
    """
    This class is extra scaffolding around a model. Modify this class to specify how the model recieves prompts and generates responses.

    Example usage:
        chatbot = Chatbot()
        response = chatbot.get_response("What options are available for me?")
    """

    def __init__(self):
        """
        Initialize the chatbot with a HF model ID
        """
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL # define MY_MODEL in config.py if you create a new model in the HuggingFace Hub
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        
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
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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
