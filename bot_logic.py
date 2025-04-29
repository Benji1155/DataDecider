# bot_logic.py

def get_bot_response(user_input):
    # This is where you plug in your Jupyter bot logic
    # For now, let's keep it simple
    if "hello" in user_input.lower():
        return "Hi there! How can I help you today?"
    elif "amount" in user_input.lower():
        return "Would you like to visualize the amount data?"
    else:
        return "I'm not sure how to respond to that. Can you rephrase?"
