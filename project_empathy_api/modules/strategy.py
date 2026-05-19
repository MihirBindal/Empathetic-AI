STRATEGY_DEFINITIONS = {
    "Question": "Asking open or closed questions to encourage the user to share more detail.",
    "Restatement or Paraphrasing": "Repeating what the user said in different words to show understanding.",
    "Reflection of Feelings": "Explicitly naming the emotions the user is expressing.",
    "Self-disclosure": "Sharing a similar personal experience to build rapport.",
    "Affirmation and Reassurance": "Validating the user's strengths or normalising their feelings.",
    "Providing Suggestions": "Offering actionable advice or new perspectives."
}

def get_strategy_desc(label):
    return STRATEGY_DEFINITIONS.get(label, "A general empathetic response strategy.")