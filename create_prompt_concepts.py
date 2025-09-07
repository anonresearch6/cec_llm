INTRO_BLURB = "Below is a clinical note. Write a response that appropriately completes the request.\n"
INSTRUCTION_KEY = "### Instruction: You will be asked whether a medical concept is relevant with the following clinical note or not. The concept will be a phrase of a few words. Write your" 	
INSTRUCTION_KEY	+= " answer in only one word and nothing else. Your replying word should be 'yes' or 'no'. The question will be : Is the note above relevant with the concept: {concept}?"
INSTRUCTION_KEY += " You will be provided an actual concept in place of {concept}.\n"

def create_concept_prompt(note, concept, flag=True, train=True):
    blurb = f"### Clinical Note :\n{note}\n"
    question = f'### Question: Is the note above relevant with the concept \'{concept}\'?\n'
    RESPONSE_KEY = "### Answer: no" if not flag else "### Answer: yes"
    if train:
        prompt = INTRO_BLURB + INSTRUCTION_KEY + blurb + question + RESPONSE_KEY
    else:
        prompt = INTRO_BLURB + INSTRUCTION_KEY + blurb + question + "### Answer:"
    return prompt