import os, sys

import argparse
import numpy as np
import openai
import pandas as pd
from tqdm import tqdm
import re
import json
import time
from fuzzywuzzy import fuzz

# Ensure you have set your OpenAI API key
client = openai.OpenAI(api_key=OPENAI_API_KEY)

CHARACTER_PROMPTS = {
    'Detective':
        """
        You are a world-weary detective narrating your investigations. Your tone is gritty, mysterious, and evocative of noir fiction. Reformulate the following questions as if you’re puzzling out clues in a case. Keep the core meaning intact but phrase it in your distinct detective style. Here are some examples to guide you:
        
        Few-Shot Examples:
            1.	Original Question: Who is Shahid Beheshti University named after?
        Detective Style: Who’s the figure behind the name of Shahid Beheshti University, huh?
            2.	Original Question: Who is Mahyeldi married to?
        Detective Style: Who’s the lucky soul tied to Mahyeldi, if you catch my drift?
            3.	Original Question: For which organization does Andreas Bergh work?
        Detective Style: What outfit’s got Andreas Bergh on the payroll?
            4.	Original Question: In which city did Pierre Dignac pass away?
        Detective Style: Pierre Dignac—what city did he take his final bow in, eh?
        
        Task:
        
        Now it’s your turn. Reformulate the following question in the same detective style:
        Original Question: {}
        Detective Style:
        """,

    'Journalist':
        """
        You are a seasoned journalist with a flair for asking sharp, professional questions to uncover the truth. Your tone is inquisitive, polished, and slightly formal, as if you’re gathering information for an important article. Reformulate the following questions as if you’re conducting an interview or writing a headline. Keep the core meaning intact but phrase it in your distinct journalist style. Here are some examples to guide you:
        
        Few-Shot Examples:
        
            1.	Original Question: Who is Shahid Beheshti University named after?
        Journalist Style: Can you identify the individual for whom Shahid Beheshti University is named?
            2.	Original Question: Who is Mahyeldi married to?
        Journalist Style: Who is recognized as Mahyeldi’s spouse?
            3.	Original Question: For which organization does Andreas Bergh work?
        Journalist Style: What organization currently employs Andreas Bergh?
            4.	Original Question: In which city did Pierre Dignac pass away?
        Journalist Style: Which city is documented as the location of Pierre Dignac’s death?
        
        Task:
        
        Now it’s your turn. Reformulate the following question in the same journalist style:
        Original Question: {}
        Journalist Style:
        """,

    'Historian':
        """
        You are a historian with a deep interest in documenting and analyzing the past. Your tone is thoughtful, precise, and scholarly, as if you’re crafting a passage for an academic work or historical record. Reformulate the following questions as if you are seeking to place the facts within a broader historical context. Keep the core meaning intact but phrase it in your distinct historian style. Here are some examples to guide you:
        
        Few-Shot Examples:
        
            1.	Original Question: Who is Shahid Beheshti University named after?
        Historian Style: Which historical figure serves as the namesake for Shahid Beheshti University?
            2.	Original Question: Who is Mahyeldi married to?
        Historian Style: Who is recorded as the spouse of Mahyeldi in the annals of personal history?
            3.	Original Question: For which organization does Andreas Bergh work?
        Historian Style: To which institution or organization is Andreas Bergh affiliated in his professional capacity?
            4.	Original Question: In which city did Pierre Dignac pass away?
        Historian Style: What city is noted in historical accounts as the place of Pierre Dignac’s death?
        
        Task:
        
        Now it’s your turn. Reformulate the following question in the same historian style:
        Original Question: {}
        Historian Style:
        """,

    'Casual':
        """
        You are a casual, friendly conversationalist with a relaxed and approachable tone. You’re curious and engaging, as if you’re chatting with a friend over coffee or in a group discussion. Reformulate the following questions to sound natural and conversational while keeping the core meaning intact. Here are some examples to guide you:
        
        Few-Shot Examples:
        
            1.	Original Question: Who is Shahid Beheshti University named after?
        Casual Conversationalist Style: Do you know who Shahid Beheshti University is named after?
            2.	Original Question: Who is Mahyeldi married to?
        Casual Conversationalist Style: So, who’s Mahyeldi married to?
            3.	Original Question: For which organization does Andreas Bergh work?
        Casual Conversationalist Style: What’s the organization Andreas Bergh works for?
            4.	Original Question: In which city did Pierre Dignac pass away?
        Casual Conversationalist Style: Hey, do you know which city Pierre Dignac died in?
        
        Task:
        
        Now it’s your turn. Reformulate the following question in the same casual conversational style:
        Original Question: {}
        Casual Conversationalist Style:
        """,

    'Pirate':
        """
        You are a swashbuckling pirate with a flair for colorful language and sea-themed metaphors. Your tone is bold, rough, and full of pirate lingo, as if you’re recounting tales from the high seas. Reformulate the following questions to sound like they’re being asked by a true pirate. Keep the core meaning intact but add a piratical twist. Here are some examples to guide you:
        
        Few-Shot Examples:
        
            1.	Original Question: Who is Shahid Beheshti University named after?
        Pirate Style: Arrr! Who be the namesake of this here Shahid Beheshti University?
            2.	Original Question: Who is Mahyeldi married to?
        Pirate Style: Tell me, matey, who’s the soul that’s tied the knot with Mahyeldi?
            3.	Original Question: For which organization does Andreas Bergh work?
        Pirate Style: What be the crew Andreas Bergh’s sailin’ with these days?
            4.	Original Question: In which city did Pierre Dignac pass away?
        Pirate Style: In what port or city did Pierre Dignac meet his final fate, arrr?
        
        Task:
        
        Now it’s your turn. Reformulate the following question in the same pirate style:
        Original Question: {}
        Pirate Style:
        """,

    'Yoda':
        """
        You are Yoda, the wise and enigmatic Jedi Master. Your tone is thoughtful, and your speech follows a unique structure where the word order often emphasizes reflection and wisdom. Reformulate the following questions as if you are Yoda, maintaining the core meaning but expressing it in your characteristic style. Here are some examples to guide you:
        
        Few-Shot Examples:
        
            1.	Original Question: Who is Shahid Beheshti University named after?
        Yoda Style: Named after who, Shahid Beheshti University is, hmm?
            2.	Original Question: Who is Mahyeldi married to?
        Yoda Style: Married to who, Mahyeldi is?
            3.	Original Question: For which organization does Andreas Bergh work?
        Yoda Style: For which organization, Andreas Bergh works, tell me you can?
            4.	Original Question: In which city did Pierre Dignac pass away?
        Yoda Style: Pass away, Pierre Dignac did, but in which city, hmm?
        
        Task:
        
        Now it’s your turn. Reformulate the following question in Yoda’s wise and unique style:
        Original Question: {}
        Yoda Style:
        """,

    'Bard':
        """
        You are a Shakespearean bard, skilled in the art of eloquent and poetic expression. Your tone is dramatic, flowing, and rich with classical language reminiscent of Elizabethan England. Reformulate the following questions as if they are lines from a grand play or poetic tale. Keep the core meaning intact, but phrase it with flourish and theatrical flair. Here are some examples to guide you:
        
        Few-Shot Examples:
        
            1.	Original Question: Who is Shahid Beheshti University named after?
        Shakespearean Bard Style: Pray, who dost lend their noble name to Shahid Beheshti University?
            2.	Original Question: Who is Mahyeldi married to?
        Shakespearean Bard Style: To whom is fair Mahyeldi bound in sacred matrimony?
            3.	Original Question: For which organization does Andreas Bergh work?
        Shakespearean Bard Style: For what house or guild doth Andreas Bergh lend his labor?
            4.	Original Question: In which city did Pierre Dignac pass away?
        Shakespearean Bard Style: In what city did Pierre Dignac shuffle off this mortal coil?
        
        Task:
        
        Now it’s your turn. Reformulate the following question in the grand and poetic style of the Shakespearean bard:
        Original Question: {}
        Shakespearean Bard Style:
        """,

    'Philosopher':
        """
        You are an ancient philosopher, deeply contemplative and wise, always pondering the greater truths of existence. Your tone is reflective, profound, and formal, as if you are crafting a dialogue or treatise on the nature of knowledge and events. Reformulate the following questions as if they are inquiries befitting philosophical discourse. Keep the core meaning intact but phrase it in your distinct, thoughtful style. Here are some examples to guide you:
        
        Few-Shot Examples:
        
            1.	Original Question: Who is Shahid Beheshti University named after?
        Ancient Philosopher Style: Upon which noble individual has the honor of Shahid Beheshti University’s name been bestowed?
            2.	Original Question: Who is Mahyeldi married to?
        Ancient Philosopher Style: To whom is Mahyeldi joined in the sacred bond of matrimony, as tradition dictates?
            3.	Original Question: For which organization does Andreas Bergh work?
        Ancient Philosopher Style: To which institution or order does Andreas Bergh render his labor and allegiance?
            4.	Original Question: In which city did Pierre Dignac pass away?
        Ancient Philosopher Style: In what city did Pierre Dignac meet the inevitable end that awaits all mortals?
        
        Task:
        
        Now it’s your turn. Reformulate the following question in the profound and reflective style of an ancient philosopher:
        Original Question: {}
        Ancient Philosopher Style:
        """,

    'Robot':
        """
        You are a highly advanced robot designed to communicate in a formal and precise manner. Your tone is technical, structured, and devoid of unnecessary emotion, reflecting your logical programming and attention to detail. Reformulate the following questions as if they are system-generated queries, maintaining clarity and a professional demeanor. Keep the core meaning intact but phrase it in your formal, robotic style. Here are some examples to guide you:
        
        Few-Shot Examples:
        
            1.	Original Question: Who is Shahid Beheshti University named after?
        Robot with Formal Speech Style: Please specify the individual associated with the naming of Shahid Beheshti University.
            2.	Original Question: Who is Mahyeldi married to?
        Robot with Formal Speech Style: Identify the individual recognized as Mahyeldi’s spouse.
            3.	Original Question: For which organization does Andreas Bergh work?
        Robot with Formal Speech Style: Provide the name of the organization that currently employs Andreas Bergh.
            4.	Original Question: In which city did Pierre Dignac pass away?
        Robot with Formal Speech Style: Indicate the city where Pierre Dignac’s death occurred.
        
        Task:
        
        Now it’s your turn. Reformulate the following question in the precise and formal style of a robot:
        Original Question: {}
        Robot with Formal Speech Style:
        """,

    'Caveman':
        """
        You are a caveman, speaking in simple and primitive language, with short sentences and limited vocabulary. Your tone is direct, straightforward, and reflects the early stages of human communication. Reformulate the following questions in a way that matches your basic and minimalistic speech style, while keeping the core meaning intact. Here are some examples to guide you:
        
        Few-Shot Examples:
        
            1.	Original Question: Who is Shahid Beheshti University named after?
        Caveman Style: Who name Shahid Beheshti University for?
            2.	Original Question: Who is Mahyeldi married to?
        Caveman Style: Who Mahyeldi mate with?
            3.	Original Question: For which organization does Andreas Bergh work?
        Caveman Style: What group Andreas Bergh work for?
            4.	Original Question: In which city did Pierre Dignac pass away?
        Caveman Style: Where Pierre Dignac die?
        
        Task:
        
        Now it’s your turn. Reformulate the following question in the simple and direct style of a caveman:
        Original Question: {}
        Caveman Style:
        """
}


def generate_open_qa_pairs(df, mode='edit', model="gpt-3.5-turbo-0125", batch_submission=False):
    # Initialize an empty list to store the results
    results = []

    # Define the prompt template
    prompt_object = """
        Given a fact triplet in the format (subject, relation, object) and a description of the relation, generate a question-answer pair where the answer is the object of the triplet. Ensure that the question explicitly references the relation and is structured to lead uniquely to the chosen answer, which must be the exact text of the object. Use the description of the relation to improve the specificity and clarity of the question. Make the question specific enough to ensure that there is only one possible correct answer, eliminating any ambiguity.
        
        Instructions for Generation:
        
            1.	Construct the Question:
            •	Ensure the question is directly tied to the relation and points exclusively to the selected answer.
            •	Include additional context from the subject to make the question more specific and unambiguous.
            •	Use the description of the relation to enhance the quality of the question.
            •	Use only the exact text of the object in the answer.
            2.	Format:
            •	Provide the question-answer pair in the following format:
            •	Q:
            •	A:
        
        Examples:
        
        Fact Triplet: Turnberry Lighthouse, color, white
        Relation Description: The color of the subject.
        Q: What is the color of Turnberry Lighthouse?
        A: white
        
        Fact Triplet: Folsom Library, part of, Rensselaer Libraries
        Relation Description: part of this subject; inverse property of “part of” (P361). See also “has parts of the class” (P2670).
        Q: What institution does Folsom Library belong to?
        A: Rensselaer Libraries
        
        Fact Triplet: Volt Europa, subsidiary, Volt Netherlands
        Relation Description: A company controlled by another company.
        Q: What is the name of the subsidiary of Volt Europa in the Netherlands?
        A: Volt Netherlands
        
        Fact Triplet: Zonia Baber, educated at, Chicago State University
        Relation Description: The institution where the subject received their education.
        Q: At which university did Zonia Baber receive her education?
        A: Chicago State University
        
        Task:
        
        According to the instructions and examples, provide a QA pair for the following fact triplet:
        
        Fact Triplet: {}, {}, {}
        Relation Description: {}
        """
    if batch_submission:
        batch = []

    # Iterate over each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        subject = row['subject']
        relation = row['relation']
        object_ = row['object']
        relation_description = row['relation_description']

        prompt = prompt_object.format(subject, relation, object_, relation_description)
        if batch_submission:
            batch.append({"custom_id": f'{mode}_{index}', "method": "POST", "url": "/v1/chat/completions",
                          "body": {"model": model, "messages": [{"role": "user", "content": prompt}],
                                   "max_tokens": 100}})
            continue
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        question, answer = process_qa_response(stream)

        # Append the result to the list
        results.append({
            'subject': subject,
            'relation': relation,
            'relation_description': relation_description,
            'object': object_,
            'question': question,
            'answer': answer
        })

    if batch_submission:
        return batch
    # Convert the results into a DataFrame
    result_df = pd.DataFrame(results)
    return result_df


def generate_generality_open_qa_pairs(df, model="gpt-3.5-turbo-0125", batch_submission=False, character=None):
    results = []

    prompt = """
    You are given a question-answer pair. Reformulate the question such that its content remains identical and the given answer is still the only accurate answer.
    
    Instructions for Generation:
    
        1.	Question Reformulation:
        •	Restructure the question while maintaining the same content and context.
        •	Ensure that the answer provided in the original question remains the only correct answer.
        2.	Response Format:
        •	Provide the reformulated question in the following format:
            •	Reformulated Question:
            
    Examples:
        Original Question: What is the capital of the United States?
        Answer: Washington, D.C.
        Reformulated Question: What city serves as the capital of the United States?
    
        Original Question: Who painted the Mona Lisa?
        Answer: Leonardo da Vinci
        Reformulated Question: By whom was the Mona Lisa painted?
    
        Original Question: When was the Declaration of Independence signed?
        Answer: July 4, 1776
        Reformulated Question: On what date was the Declaration of Independence signed?
        
    Task:
        Original Question: {}
        Answer: {}
    """

    if batch_submission:
        batch = []

    characters = ['Detective', 'Casual', 'Pirate', 'Philosopher', 'Caveman']

    for index in tqdm(df.index):
        question = df.loc[index, 'question']
        answer = df.loc[index, 'answer']

        if character is not None:
            if character == 'mixed':
                c = characters[np.random.randint(5)]
            else:
                c = character
            prompt_qa = CHARACTER_PROMPTS[c].format(question)
        else:
            prompt_qa = prompt.format(question, answer)
        if batch_submission:
            batch.append({"custom_id": f'gen{2 if character else ""}_{index}', "method": "POST", "url": "/v1/chat/completions",
                          "body": {"model": model, "messages": [{"role": "user", "content": prompt_qa}],
                                   "max_tokens": 100}})
            continue
        try_ = 0
        while try_ <= 3:
            try:
                try_ += 1
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt_qa}],
                    stream=True,
                )
                response = "".join(
                    [chunk.choices[0].delta.content if chunk.choices[0].delta.content is not None else '' for chunk in stream])
                break
            except Exception as e:
                print(f'Error: {e}')
                time.sleep(5)
                continue

        # Split the generated text into question and answer
        qa_pair = response.split("\n")
        if len(qa_pair) == 1:
            reformulated_question = qa_pair[0].replace("Reformulated Question:", "").strip()
        else:
            reformulated_question = ""

        results.append({
            'question': question,
            'reformulated_question': reformulated_question,
            'answer': answer,
        })

    if batch_submission:
        return batch

    result_df = pd.DataFrame(results)
    return result_df


def generate_mhop_qa_pairs(df, model="gpt-3.5-turbo-0125", batch_submission=False):
    # Initialize an empty list to store the results
    results = []

    # Define the prompt template
    prompt_open = """
        Given a multi-hop fact tuple in the format (entity_0, relation_1, entity_1, relation_2, entity_2) and a description of the relation, generate a multi-hop question-answer pair where the answer is entity_2 of the tuple. Ensure that the question explicitly references the relations and is structured to lead uniquely to the chosen answer, which must be the exact text of entity_2. Use the description of the relation to improve the specificity and clarity of the question. Make the question specific enough to ensure that there is only one possible correct answer, eliminating any ambiguity.

        Instructions for Generation:

            1.	Construct the Question:
            •	Ensure the question uses the relations to ask a multi-hop question leading to the selected answer.
            •   Ensure that the question does not contains the answer or the object of the first relation i.e. entity_1.
            •	Include additional context from the subject to make the question more specific and unambiguous.
            •	Use the description of the relation to enhance the quality of the question.
            •	Use only the exact text of the entity_2 in the answer.
            2.	Format:
            •	Provide the question-answer pair in the following format:
            •	Q:
            •	A:

        Examples:

        Fact Tuple: USA, President, Joe Biden, Son, Hunter Biden
        Relation Descriptions: President of this country., Son of this person.
        Q: Who is the son of the President of the USA?
        A: Hunter Biden

        Fact Tuple: Fairfield Offices, historic county, Lanarkshire, country, Scotland
        Relation Descriptions: Historic county of this place., Country of this place.
        Q: In which country is the historic county of the Fairfield Offices located?
        A: Scotland
        
        Fact Tuple: Josef Novak, country of citizenship, Hungary, highest point, Kékes
        Relation Descriptions: Country of citizenship of this person., Highest point of this place.
        Q: What is the highest point of the country of citizenship of Josef Novak?
        A: Kékes
        
        Fact Tuple: The German Doctor, significant event, 86th Academy Awards, location, Dolby Theatre
        Relation Descriptions: Significant event in this place., Location of this event.
        Q: Where did the significant event tied to "The German Doctor" take place?
        A: Dolby Theatre

        Task:

        According to the instructions and examples, provide a multi hop QA pair for the following fact tuple:

        Fact Tuple: {}, {}, {}, {}, {}
        Relation Descriptions: {}, {}
        """

    prompt_open_v2 = """
            Given a multi-hop fact tuple in the format (entity_0, relation_1, entity_1, relation_2, entity_2) and a description of the relation, generate a multi-hop question-answer pair where the answer is entity_2 of the tuple. Ensure that the question explicitly references the relations and is structured to lead uniquely to the chosen answer, which must be the exact text of entity_2. Use the description of the relation to improve the specificity and clarity of the question. Make the question specific enough to ensure that there is only one possible correct answer, eliminating any ambiguity.

            Instructions for Generation:

                1.	Construct the Question:
                •	Ensure the question uses the relations to ask a multi-hop question leading to the selected answer.
                •	Include additional context from the subject to make the question more specific and unambiguous.
                •	Use the description of the relation to enhance the quality of the question.
                •	Use only the exact text of the entity_2 in the answer.
                2.	Format:
                •	Provide the question-answer pair in the following format:
                •	Q:
                •	A:

            Examples:

            Fact Tuple: USA, President, Entity, Son, Hunter Biden
            Relation Descriptions: President of this country., Son of this person.
            Q: Who is the son of the President of the USA?
            A: Hunter Biden

            Fact Tuple: Fairfield Offices, historic county, Entity, country, Scotland
            Relation Descriptions: Historic county of this place., Country of this place.
            Q: In which country is the historic county of the Fairfield Offices located?
            A: Scotland

            Fact Tuple: Josef Novak, country of citizenship, Entity, highest point, Kékes
            Relation Descriptions: Country of citizenship of this person., Highest point of this place.
            Q: What is the highest point of the country of citizenship of Josef Novak?
            A: Kékes

            Fact Tuple: The German Doctor, significant event, Entity, location, Dolby Theatre
            Relation Descriptions: Significant event in this place., Location of this event.
            Q: Where did the significant event tied to "The German Doctor" take place?
            A: Dolby Theatre

            Task:

            According to the instructions and examples, provide a multi hop QA pair for the following fact tuple:

            Fact Tuple: {}, {}, {}, {}, {}
            Relation Descriptions: {}, {}
            """

    prompt_open_v3 = """
        Given a multi-hop fact tuple in the format (entity_0, relation_1, entity_1, relation_2, entity_2) and a description of each relation, generate a multi-hop question-answer pair. Ensure that the answer is “entity_2” and not “entity_1.” Structure the question to specifically query about “entity_2” in a way that leads uniquely to this as the answer. The question should reference both relations, making it specific enough to ensure only one possible correct answer—entity_2.

        Instructions for Generation:
        
            1.	Construct the Question:
            •	Design the question to focus on identifying entity_2 by logically combining relation_1 and relation_2.
            •	Include context from the subject (entity_0) to make the question unambiguous.
            •	Use the relation descriptions to improve clarity and focus of the question.
            •	Ensure the question targets entity_2 as the answer, avoiding phrasing that might prompt entity_1 instead.
            2.	Format:
            •	Structure the output in the following format:
            •	Q: [Constructed Question]
            •	A: [entity_2]
        
        Examples:
        
        Fact Tuple: USA, President, Joe Biden, Son, Hunter Biden
        Relation Descriptions: President of this country., Son of this person.
        Q: Who is the son of the President of the USA?
        A: Hunter Biden
        
        Fact Tuple: Josef Novak, country of citizenship, Hungary, highest point, Kékes
        Relation Descriptions: Country of citizenship of this person., Highest point of this place.
        Q: What is the highest point of the country of citizenship of Josef Novak?
        A: Kékes
        
        Fact Tuple: Pholidaster squamatus, taxon author, Percy Sladen, field of work, starfish
        Relation Descriptions: Taxon author of this species., Field of work of this person.
        Q: What is the field of work of the taxon author of Pholidaster squamatus?
        A: Starfish
        
        Task:
        
        According to these instructions and examples, provide a multi-hop QA pair for the following fact tuple:
        
        Fact Tuple: {}, {}, {}, {}, {}
        Relation Descriptions: {}, {}
    """

    prompt_open_v4 = """
        Given a multi-hop fact tuple in the format (entity_0, relation_1, [MASKED_ENTITY_1], relation_2, entity_2) and a description of each relation, generate a multi-hop question-answer pair. Ensure that the answer is “entity_2” and that [MASKED_ENTITY_1] is not referenced in the question or answer. Structure the question to specifically query about entity_2 in a way that logically integrates relation_1 and relation_2 while leading uniquely to entity_2 as the answer.
    
        Instructions for Generation:
        
            1.	Construct the Question:
            •	Design the question to identify entity_2 by logically combining relation_1 and relation_2.
            •	Include context from the subject (entity_0) to make the question unambiguous.
            •	Use the relation descriptions to improve clarity and focus of the question.
            •	Avoid any mention of [MASKED_ENTITY_1] in the question or as a possible answer, ensuring the question uniquely targets entity_2.
            2.	Format:
            •	Structure the output in the following format:
            •	Q: [Constructed Question]
            •	A: [entity_2]
        
        Examples:
        
        Fact Tuple: USA, President, [MASKED_ENTITY_1], Son, Hunter Biden
        Relation Descriptions: President of this country., Son of this person.
        Q: Who is the son of the President of the USA?
        A: Hunter Biden
        
        Fact Tuple: Josef Novak, country of citizenship, [MASKED_ENTITY_1], highest point, Kékes
        Relation Descriptions: Country of citizenship of this person., Highest point of this place.
        Q: What is the highest point of the country of citizenship of Josef Novak?
        A: Kékes
        
        Fact Tuple: Pholidaster squamatus, taxon author, [MASKED_ENTITY_1], field of work, starfish
        Relation Descriptions: Taxon author of this species., Field of work of this person.
        Q: What is the field of work of the taxon author of Pholidaster squamatus?
        A: Starfish
        
        Task:
        
        According to these instructions and examples, provide a multi-hop QA pair for the following fact tuple:
        
        Fact Tuple: {}, {}, [MASKED_ENTITY_1], {}, {}
        Relation Descriptions: {}, {}
    """

    prompt_open_v5 = """
        Given a multi-hop fact tuple in the format (entity_0, relation_1, [MASKED_ENTITY_1], relation_2, entity_2) and a description of each relation, generate a multi-hop question-answer pair where the answer is “entity_2.” Ensure that the question respects the directional nature of each relation, especially relation_2, so that it logically reflects how the masked entity_1 relates to entity_2. The question should be structured to uniquely lead to entity_2 as the answer.
    
        Instructions for Generation:
        
            1.	Construct the Question:
            •	Design the question to identify entity_2 by logically combining relation_1 and relation_2.
            •	Respect the directionality of each relation as described: if relation_2 describes a relation where [MASKED_ENTITY_1] is the subject and entity_2 is the object (e.g., “is a member of”), ensure this direction is reflected in the question.
            •	Avoid reversing the direction of relation_2, and confirm the natural flow between [MASKED_ENTITY_1] and entity_2.
            •	Include context from entity_0 to make the question clear and specific.
            •	Do not reference [MASKED_ENTITY_1] in the answer; only entity_2 should be the answer.
            2.	Format:
            •	Provide the output in the following format:
            •	Q: [Constructed Question]
            •	A: [entity_2]
        
        Examples:
        
        Fact Tuple: Yasis Malih, place of birth, [MASKED_ENTITY_1], member of, Organization of World Heritage Cities
        Relation Descriptions: Place of birth of this person., Membership of this place.
        Q: Which organization is the place of birth of Yasis Malih a member of?
        A: Organization of World Heritage Cities
        
        Fact Tuple: Intelligent Land Investments, instance of, [MASKED_ENTITY_1], owned by, entrepreneur
        Relation Descriptions: Instance type of this entity., Owner of this type.
        Q: Who owns the instance of Intelligent Land Investments?
        A: Entrepreneur
        
        Fact Tuple: Josef Novak, country of citizenship, [MASKED_ENTITY_1], highest point, Kékes
        Relation Descriptions: Country of citizenship of this person., Highest point of this place.
        Q: What is the highest point of the country of citizenship of Josef Novak?
        A: Kékes
        
        Fact Tuple: Pholidaster squamatus, taxon author, [MASKED_ENTITY_1], field of work, starfish
        Relation Descriptions: Taxon author of this species., Field of work of this person.
        Q: What is the field of work of the taxon author of Pholidaster squamatus?
        A: Starfish
        
        Task:
        
        According to these instructions and examples, provide a multi-hop QA pair for the following fact tuple:
        
        Fact Tuple: {}, {}, [MASKED_ENTITY_1], {}, {}
        Relation Descriptions: {}, {}
    """

    prompt_correct = """
        Given a multi-hop fact tuple in the format (entity_0, relation_1, entity_1, relation_2, entity_2) and a description of the relation, generate a question about a factual statement where the response is always “correct”. Use the description of the relation to improve the specificity and clarity of the statement.

        Instructions for Generation:

            1.	Construct the Statement:
            •	Ensure the statement is directly tied to the relations and points exclusively to the selected answer.
            •	Include additional context from the subject to make the statement more specific and unambiguous.
            •	Use the description of the relations to enhance the quality of the statement.
            •	Ensure the response is either “correct”.
            2.	Format:
            •	Provide the question and response in the following format:
            •	Q: [Question]
            •	A: correct

        Examples:

        Fact Tuple: USA, President, Jow Biden, Son, Hunter Biden
        Relation Descriptions: President of this country., Son of this person.
        Q: Is it correct to say that Hunter Biden is the son of the President of the USA?
        A: correct

        Fact Tuple: Fairfield Offices, historic county, Lanarkshire, country, Scotland
        Relation Descriptions: Historic county of this place., Country of this place.
        Q: Is it accurate to say that the historic county of the Fairfield Offices is located in Scotland?
        A: correct

        Fact Tuple: Umbrella, composer, Rihanna, unmarried partner, A$AP Rocky
        Relation Descriptions: Composer of this song., Unmarried partner of this person.
        Q: Is it correct that A$AP Rocky is the unmarried partner of the composer of the song "Umbrella"?
        A: correct

        Task:

        According to the instructions and examples, provide a QA pair for the following fact triplet:

        Fact Tuple: {}, {}, {}, {}, {}
        Relation Descriptions: {}, {}
        """
    prompt_incorrect = """
        Given a multi-hop fact tuple in the format (entity_0, relation_1, entity_1, relation_2, entity_2) and a description of the relation, generate a question about a factual statement where the response is always “incorrect”. Use the description of the relation to improve the specificity and clarity of the statement.

        Instructions for Generation:

            1.	Construct the Statement:
            •	Ensure the statement is directly tied to the relations and points exclusively to the selected answer.
            •	Include additional context from the subject to make the statement more specific and unambiguous.
            •	Use the description of the relations to enhance the quality of the statement.
            •	Ensure the response is either “incorrect”.
            2.	Format:
            •	Provide the question and response in the following format:
            •	Q: [Question]
            •	A: correct

        Examples:

        Fact Tuple: USA, President, Jow Biden, Son, Hunter Biden
        Relation Descriptions: President of this country., Son of this person.
        Q: Is it correct to say that Hunter Biden is not the son of the President of the USA?
        A: incorrect

        Fact Tuple: Fairfield Offices, historic county, Lanarkshire, country, Scotland
        Relation Descriptions: Historic county of this place., Country of this place.
        Q: Is it accurate to say that the historic county of the Fairfield Offices is not located in Scotland?
        A: incorrect

        Fact Tuple: Umbrella, composer, Rihanna, unmarried partner, A$AP Rocky
        Relation Descriptions: Composer of this song., Unmarried partner of this person.
        Q: Is it correct that A$AP Rocky is not the unmarried partner of the composer of the song "Umbrella"?
        A: incorrect

        Task:

        According to the instructions and examples, provide a QA pair for the following fact triplet:

        Fact Tuple: {}, {}, {}, {}, {}
        Relation Descriptions: {}, {}
        """

    if batch_submission:
        batch = []

    # Iterate over each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        subject = row['subject']
        relation = row['relation']
        object_ = row['object']
        relation_description = row['relation_description']
        mhop_relation = row['mhop_relation']
        mhop_object = row['mhop_object']
        mhop_relation_description = row['mhop_relation_description']

        if mhop_object == '':
            results.append({
                'subject': subject,
                'relation': relation,
                'relation_description': relation_description,
                'object': object_,
                'mhop_relation': mhop_relation,
                'mhop_object': mhop_object,
                'question': '',
                'answer': ''
            })
            continue

        if row['open']:
            #prompt = prompt_open_v3.format(subject, relation, object_, mhop_relation, mhop_object, relation_description, mhop_relation_description)
            prompt = prompt_open_v5.format(subject, relation, mhop_relation, mhop_object, relation_description,
                                           mhop_relation_description)
        else:
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                prompt = prompt_correct.format(subject, relation, object_, mhop_relation, mhop_object,
                                               relation_description, mhop_relation_description)
            else:
                prompt = prompt_incorrect.format(subject, relation, object_, mhop_relation, mhop_object,
                                                 relation_description, mhop_relation_description)
        if batch_submission:
            batch.append({"custom_id": f'mhop_{index}', "method": "POST", "url": "/v1/chat/completions",
                          "body": {"model": model, "messages": [{"role": "user", "content": prompt}],
                                   "max_tokens": 100}})
            continue
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        question, answer = process_qa_response(stream)

        # Append the result to the list
        results.append({
            'subject': subject,
            'relation': relation,
            'relation_description': relation_description,
            'object': object_,
            'mhop_relation': mhop_relation,
            'mhop_object': mhop_object,
            'question': question,
            'answer': answer
        })

    if batch_submission:
        return batch

    # Convert the results into a DataFrame
    result_df = pd.DataFrame(results)
    return result_df


def process_qa_response(stream):
    if isinstance(stream, dict):
        response = stream['body']['choices'][0]['message']['content']
    else:
        response = "".join(
            [chunk.choices[0].delta.content if chunk.choices[0].delta.content is not None else '' for chunk in stream])

    # Split the generated text into question and answer
    question, answer = "", ""
    qa_pair = response.split("\n")
    for element in qa_pair:
        if element.startswith("Q:"):
            question = element.replace("Q:", "").strip()
        elif element.startswith("A:"):
            answer = element.replace("A:", "").strip()
        elif element.startswith("Reformulated Question:"):
            question = element.replace("Reformulated Question:", "").strip()
        elif "Style:" in element:
            question = element.split("Style:")[-1].strip()
        else:
            question = element.strip()
    return question, answer


def contains_answer_in_question(row):
    question = row['question']
    answer = row['answer']

    # Split the answer into individual items if it's a list
    answer_items = [item.strip() for item in answer.split(',')]

    # Check if any of the answer items are in the question
    for item in answer_items:
        if item in question:
            return True
    return False


def filter_questions(edit, locality, generality, personas):
    if not len(edit) == len(locality) == len(generality) == len(personas):
        ids_edit = edit['id'].tolist()
        ids_locality = locality['id'].tolist()
        ids_generality = generality['id'].tolist()
        for df in [edit, locality, generality, personas]:
            df.drop(df[~df['id'].isin(ids_edit) | ~df['id'].isin(ids_locality) | ~df['id'].isin(ids_generality)].index,
                    inplace=True)
    assert len(edit) == len(locality) == len(generality) == len(personas), \
        f"The number of questions in each set should be the same ({len(edit)}, {len(locality)}, {len(generality)}, {len(personas)})"
    print("Filtering questions")
    print("Original number of questions: ", len(edit))

    idxs_to_drop = []
    for i in range(len(edit)):
        # check if any question of answer is empty
        if (edit.iloc[i]['question'] == '' or edit.iloc[i]['answer'] == '' or
                locality.iloc[i]['question'] == '' or locality.iloc[i]['answer'] == '' or
                generality.iloc[i]['question'] == '' or generality.iloc[i]['answer'] == '' or
                personas.iloc[i]['question'] == '' or personas.iloc[i]['answer'] == ''):
            idxs_to_drop.append(i)
            continue

        # check if any question of answer is nan
        if (pd.isna(edit.iloc[i]['question']) or pd.isna(edit.iloc[i]['answer']) or
            pd.isna(locality.iloc[i]['question']) or pd.isna(locality.iloc[i]['answer']) or
            pd.isna(generality.iloc[i]['question']) or pd.isna(generality.iloc[i]['answer'])
            or pd.isna(personas.iloc[i]['question']) or pd.isna(personas.iloc[i]['answer'])):
            idxs_to_drop.append(i)
            continue

        contains_nan = False
        for qa in [edit, locality, generality, personas]:
            if qa is None:
                continue
            for col in ['question', 'answer', 'subject', 'object', 'relation']:
                # check for isna and isnull
                if pd.isna(qa.iloc[i][col]) or pd.isnull(qa.iloc[i][col]) or isinstance(qa.iloc[i][col], float):
                    contains_nan = True
                    break
        if contains_nan:
            idxs_to_drop.append(i)
            continue

        # check if all questions contain the subject
        edit_contains_subject = contains(edit.iloc[i]['subject'], edit.iloc[i]['question'])
        locality_contains_subject = contains(locality.iloc[i]['subject'], locality.iloc[i]['question'])
        generality_contains_subject = contains(generality.iloc[i]['subject'], generality.iloc[i]['question'])
        personas_contains_subject = contains(personas.iloc[i]['subject'], personas.iloc[i]['question'])

        if not edit_contains_subject or not locality_contains_subject or not generality_contains_subject or not personas_contains_subject:
            idxs_to_drop.append(i)
            continue

        # check if all answers are identical to the objects
        edit_answer_check = contains(edit.iloc[i]['answer'], edit.iloc[i]['object'])
        locality_answer_check = contains(locality.iloc[i]['answer'], locality.iloc[i]['object'])

        if not edit_answer_check or not locality_answer_check:
            idxs_to_drop.append(i)
            continue

        edit_contains_answer = contains_answer_in_question(edit.iloc[i])
        locality_contains_answer = contains_answer_in_question(locality.iloc[i])
        generality_contains_answer = contains_answer_in_question(generality.iloc[i])
        personas_contains_answer = contains_answer_in_question(personas.iloc[i])

        if edit_contains_answer or locality_contains_answer or generality_contains_answer or personas_contains_answer:
            idxs_to_drop.append(i)

    print("Number of questions after filtering: ", len(edit) - len(idxs_to_drop))
    edit = edit.drop(idxs_to_drop).reset_index(drop=True)
    locality = locality.drop(idxs_to_drop).reset_index(drop=True)
    generality = generality.drop(idxs_to_drop).reset_index(drop=True)
    personas = personas.drop(idxs_to_drop).reset_index(drop=True)

    return edit, locality, generality, personas


def filter_mhop_qa(mhop, edit_filtered):
    edit_tripltes = [(row['subject'], row['relation'], row['object']) for i, row in edit_filtered.iterrows()]
    for i, row in mhop.iterrows():
        if row['mhop_object'].lower() in row['question'].lower():
            mhop.drop(i, inplace=True)
        elif (row['subject'], row['relation'], row['object']) not in edit_tripltes:
            mhop.drop(i, inplace=True)
        elif (row['object'], row['mhop_relation'], row['mhop_object']) not in edit_tripltes:
            mhop.drop(i, inplace=True)
    return mhop


def contains(text, question):
    try:
        text = text.replace('\'s', '')
    except AttributeError:
        print('error at:', text)
        # print type of object that raised the error
        print(type(text))
        raise AttributeError
    # Remove any punctuation from the object
    text = re.sub(r'[^\w\s]', '', text)
    # Split the object into individual words and convert them to lowercase
    object_words = set(text.lower().split())
    # Remove \'s from the question
    question = question.replace('\'s', '')
    # Remove any punctuation from the question
    question = re.sub(r'[^\w\s]', '', question)
    # Convert the question to lowercase and split into words
    question_words = set(question.lower().split())
    # Check if all words in the object are in the question
    return object_words.issubset(question_words)


def job_metadata_to_dict(job):
    return {
        "id": job.id,
        "object": job.object,
        "endpoint": job.endpoint,
        "errors": job.errors.data if job.errors is not None else None,
        "input_file_id": job.input_file_id,
        "completion_window": job.completion_window,
        "status": job.status,
        "output_file_id": job.output_file_id,
        "error_file_id": job.error_file_id,
        "created_at": job.created_at,
        "in_progress_at": job.in_progress_at,
        "expires_at": job.expires_at,
        "completed_at": job.completed_at,
        "failed_at": job.failed_at,
        "expired_at": job.expired_at,
        "request_counts": {
            "total": job.request_counts.total,
            "completed": job.request_counts.completed,
            "failed": job.request_counts.failed,
        }
    }


def submit_batch_request(directory, mode='open', model="gpt-3.5-turbo"):
    # get all submission files in directory
    files = os.listdir(directory + '/sub')
    files = [f for f in files if 'submission' in f]

    assert len(files) > 0, "No submission files found in directory"

    for i, file in enumerate(files):
        file_created = False
        retry = 0
        while not file_created and retry < 5:
            try:
                batch_input_file = client.files.create(
                    file=open(f"{directory}/sub/{file}", "rb"),
                    purpose="batch"
                )
                file_created = True
            except Exception as e:
                print(f"Error creating file: {e}")
                time.sleep(60)
                retry += 1

        batch_input_file_id = batch_input_file.id
        #check = input(f"Submit batch job for {mode} questions? (y/n): ")
        #if check.lower() != 'y':
        #    sys.exit()
        job_submitted = False
        retry = 0
        while not job_submitted and retry < 5:
            try:
                job = client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": "qa generation test"
                    }
                )
                job_submitted = True
            except Exception as e:
                print(f"Error submitting batch job: {e}")
                time.sleep(60)
                retry += 1

        status = client.batches.retrieve(job.id)

        print(f"Batch job submitted with ID: {job.id}")
        print(f"Status: {status}")

        # save the job metadata to a file
        os.makedirs(f"{directory}/{job.id}", exist_ok=True)
        with open(f"{directory}/{job.id}/{file}_metadata.json", "w") as f:
            json.dump(job_metadata_to_dict(job), f)

        done = False
        start_time = time.time()
        while not done:
            status = client.batches.retrieve(job.id)
            if status.status == 'completed':
                done = True
                print(
                    f"[{round(i / len(files), 2)}] Batch job with ID {job.id} completed in {time.time() - start_time} seconds")
            elif status.status == 'failed':
                print(f"[{round(i / len(files), 2)}] Batch job with ID {job.id} failed")
                sys.exit()
            else:
                print(
                    f"[{round(i / len(files), 2)}] Batch job with ID {job.id} still in progress - runtime: {time.time() - start_time} seconds")
                time.sleep(60)


def retrieve_batch_results(directory, job_ids):
    edit_df, loc_df, gen_df, gen2_df = [], [], [], []
    for job_id in job_ids:
        job = client.batches.retrieve(job_id)

        print(f"Retrieving batch job with ID: {job_id}")
        print(f"Status: {job.status}")

        # download the output file
        try:
            output_file = client.files.content(job.output_file_id)
        except Exception as e:
            print(f"Error downloading output file: {e}")
            continue
        with open(f"{directory}/{job_id}/output.jsonl", "wb") as f:
            f.write(output_file.content)

        edit_q, edit_a, edit_id = [], [], []
        loc_q, loc_a, loc_id = [], [], []
        gen_q, gen_a, gen_id = [], [], []
        gen2_q, gen2_a, gen2_id = [], [], []
        for out in output_file.content.decode().split('\n'):
            if out == '':
                continue
            out = eval(out.replace('null', 'None'))
            q, a = process_qa_response(out['response'])
            if 'edit' in out['custom_id']:
                edit_q.append(q)
                edit_a.append(a)
                edit_id.append(out['custom_id'].split('_')[-1])
            elif 'loc' in out['custom_id']:
                loc_q.append(q)
                loc_a.append(a)
                loc_id.append(out['custom_id'].split('_')[-1])
            elif 'gen2' in out['custom_id']:
                gen2_q.append(q)
                gen2_a.append(a)
                gen2_id.append(out['custom_id'].split('_')[-1])
            elif 'gen' in out['custom_id']:
                gen_q.append(q)
                gen_a.append(a)
                gen_id.append(out['custom_id'].split('_')[-1])
            else:
                raise ValueError(f"Invalid custom_id: {out['custom_id']}")

        edit_df.append(
            pd.DataFrame({'question': edit_q, 'answer': edit_a, 'id': edit_id}).sort_values('id').reset_index(
                drop=True)) if len(edit_q) > 0 else None
        loc_df.append(pd.DataFrame({'question': loc_q, 'answer': loc_a, 'id': loc_id}).sort_values('id').reset_index(
            drop=True)) if len(loc_q) > 0 else None
        gen_df.append(pd.DataFrame({'question': gen_q, 'answer': gen_a, 'id': gen_id}).sort_values('id').reset_index(
            drop=True)) if len(gen_q) > 0 else None
        gen2_df.append(
            pd.DataFrame({'question': gen2_q, 'answer': gen2_a, 'id': gen2_id}).sort_values('id').reset_index(
                drop=True)) if len(gen2_q) > 0 else None

    edit_df = pd.concat(edit_df) if len(edit_df) > 0 else None
    loc_df = pd.concat(loc_df) if len(loc_df) > 0 else None
    gen_df = pd.concat(gen_df) if len(gen_df) > 0 else None
    gen2_df = pd.concat(gen2_df) if len(gen2_df) > 0 else None

    edit_df.to_csv(f"{directory}/edit_unfiltered.csv", index=False, errors='replace') if edit_df is not None else None
    loc_df.to_csv(f"{directory}/loc_unfiltered.csv", index=False, errors='replace') if loc_df is not None else None
    gen_df.to_csv(f"{directory}/gen_unfiltered.csv", index=False, errors='replace') if gen_df is not None else None
    gen2_df.to_csv(f"{directory}/gen2_unfiltered.csv", index=False, errors='replace') if gen2_df is not None else None


def retrieve_batch_status(job_ids):
    status = []
    for job_id in job_ids:
        job = client.batches.retrieve(job_id)
        status.append(job.status == 'completed')
        print(f"Batch job with ID {job_id} status: {job.status}")
    return 'completed' if all(status) else 'in progress'


def check_batch(path):
    jobs = os.listdir(path)
    if len(jobs) > 0:
        print("Batch jobs already submitted")

        # get status of latest job
        job_ids = [j for j in jobs if 'batch' in j]

        status = retrieve_batch_status(job_ids)

        if status == 'completed':
            print(f"Retrieving batch results for jobs {job_ids}")
            retrieve_batch_results(path, job_ids)
        else:
            print(f"Jobs {job_ids} are still in progress")
            sys.exit()


def save_submission_file(submission, path):
    os.makedirs(path + "/sub", exist_ok=True)
    num_subs = len(submission) // 500 + 1
    for i in tqdm(range(0, num_subs)):
        f = open(f"{path}/sub/submission_{i}.jsonl", "w")
        for sub in submission[i * 500:min((i + 1) * 500, len(submission))]:
            json_line = json.dumps(sub)
            f.write(json_line + '\n')
        f.close()
    print(f"Submission files saved to {path}/sub")


def generate_qa_sets_batch(old, new, model, test_subset=False, pers_model=None, mhop_model=None):
    path = f"fact_qa_sets/{old}_{new}/batch_jobs/{model}"
    subset = '_subset' if test_subset else ''
    if os.path.exists(path + '/mhop/mhop_unfiltered.csv'):
        print("----------- Filter mhop set -----------")
        mhop = pd.read_csv(path + '/mhop/mhop_unfiltered.csv').sort_values('id').reset_index(drop=True)
        edit = pd.read_csv(f"fact_qa_sets/{old}_{new}/edit_{model}{subset}.csv")
        mhop = filter_mhop_qa(mhop, edit)
        mhop.to_csv(path + f'edit_{model}{subset}.csv', index=False)
        return "done"

    if os.path.exists(f"fact_qa_sets/{old}_{new}/edit_{model}{subset}.csv"):
        print("----------- Generate mhop set -----------")
        if os.path.exists(f"{path}/mhop"):
            print("-- Retrieve mhop batch results")
            check_batch(path + '/mhop')
            return
        print("-- Generate mhop set")
        mhop = pd.read_csv(f"fact_triplets_processed/{old}_{new}/mhop.csv")
        mhop_qa = generate_mhop_qa_pairs(mhop, model=mhop_model if mhop_model else model, batch_submission=True)
        save_submission_file(mhop_qa, f"fact_qa_sets/{old}_{new}/batch_jobs/{model}/mhop")
        submit_batch_request(path + '/mhop', model=model)
        return

    if os.path.exists(path + '/pers/pers_unfiltered.csv'):
        print("----------- QA sets already generated; Filter QA and save -----------")
        # filter qa and save to csv
        edit_qa = pd.read_csv(path + '/edit/edit_unfiltered.csv').sort_values('id').reset_index(drop=True)
        locality_qa = pd.read_csv(path + '/loc/loc_unfiltered.csv').sort_values('id').reset_index(drop=True)
        generality_qa = pd.read_csv(path + '/gen/gen_unfiltered.csv').sort_values('id').reset_index(drop=True)
        personas_qa = pd.read_csv(path + '/pers/pers_unfiltered.csv').sort_values('id').reset_index(drop=True)
        changed = pd.read_csv(f"fact_triplets_processed/{old}_{new}/changed.csv")
        locality = pd.read_csv(f"fact_triplets_processed/{old}_{new}/locality_reord.csv")
        if test_subset:
            np.random.seed(42)  # Temporary for development
            ids = np.random.choice(changed.index, 10, replace=False)
            edit = changed.loc[ids].reset_index(drop=True)
            locality = locality.loc[ids].reset_index(drop=True)
        else:
            edit = changed
        edit['question'] = edit_qa['question']
        edit['answer'] = edit_qa['answer']
        edit['id'] = edit_qa['id']
        locality['question'] = locality_qa['question']
        locality['answer'] = locality_qa['answer']
        locality['id'] = locality_qa['id']
        generality = edit.copy()
        generality['reformulated_question'] = generality_qa['question']
        generality['id'] = generality_qa['id']
        personas = edit.copy()
        personas['question'] = personas_qa['question']
        personas['id'] = personas_qa['id']
        edit, locality, generality, _ = filter_questions(edit, locality, generality, personas)
        subset = '_subset' if test_subset else ''
        edit.to_csv(f"fact_qa_sets/{old}_{new}/edit_{model}{subset}.csv", index=False)
        locality.to_csv(f"fact_qa_sets/{old}_{new}/locality_{model}{subset}.csv", index=False)
        generality.to_csv(f"fact_qa_sets/{old}_{new}/generality_{model}{subset}.csv", index=False)
        personas.to_csv(f"fact_qa_sets/{old}_{new}/personas_{model}{subset}.csv", index=False)
        return

    if os.path.exists(path + '/edit/edit_unfiltered.csv') and os.path.exists(path + '/loc/loc_unfiltered.csv') and \
        os.path.exists(path + '/gen/gen_unfiltered.csv'):
        print("----------- Generate personas set -----------")
        if os.path.exists(f"{path}/pers"):
            print("-- Retrieve generality batch results")
            check_batch(path + '/pers')
            return
        print("-- Generate personas set")
        edit_qa = pd.read_csv(path + '/edit/edit_unfiltered.csv')
        edit_qa = edit_qa.sort_values('id').reset_index(drop=True)
        changed = pd.read_csv(f"fact_triplets_processed/{old}_{new}/changed.csv")
        if test_subset:
            np.random.seed(42)  # TODO: Temporary for development
            ids = np.random.choice(changed.index, 10, replace=False)
            edit = changed.loc[ids].reset_index(drop=True)
        else:
            edit = changed
        edit['question'] = edit_qa['question']
        edit['answer'] = edit_qa['answer']
        generality_qa = generate_generality_open_qa_pairs(edit, model=pers_model if pers_model else model, batch_submission=True, character=['mixed'])
        save_submission_file(generality_qa, f"fact_qa_sets/{old}_{new}/batch_jobs/{model}/pers")
        submit_batch_request(path + '/pers', model=model)
        return

    if os.path.exists(path + '/edit/edit_unfiltered.csv') and os.path.exists(path + '/loc/loc_unfiltered.csv'):
        print("----------- Generate generality set -----------")
        if os.path.exists(f"{path}/gen"):
            print("-- Retrieve generality batch results")
            check_batch(path + '/gen')
            return
        print("-- Generate generality set")
        edit_qa = pd.read_csv(path + '/edit/edit_unfiltered.csv')
        edit_qa = edit_qa.sort_values('id').reset_index(drop=True)
        changed = pd.read_csv(f"fact_triplets_processed/{old}_{new}/changed.csv")
        locality = pd.read_csv(f"fact_triplets_processed/{old}_{new}/locality_reord.csv")
        if test_subset:
            np.random.seed(42)  # TODO: Temporary for development
            ids = np.random.choice(changed.index, 10, replace=False)
            edit = changed.loc[ids].reset_index(drop=True)
            locality = locality.loc[ids].reset_index(drop=True)
        else:
            edit = changed
        edit['question'] = edit_qa['question']
        edit['answer'] = edit_qa['answer']
        generality_qa = generate_generality_open_qa_pairs(edit, model=model, batch_submission=True)
        save_submission_file(generality_qa, f"fact_qa_sets/{old}_{new}/batch_jobs/{model}/gen")
        submit_batch_request(path + '/gen', model=model)
        return

    if os.path.exists(path + '/edit') and os.path.exists(path + '/loc'):
        print("----------- Retrieve edit batch results -----------")
        check_batch(path + '/edit')
        print("----------- Retrieve loc batch results -----------")
        check_batch(path + '/loc')
        return

    print("----------- Generate QA sets -----------")
    changed = pd.read_csv(f"fact_triplets_processed/{old}_{new}/changed.csv")
    locality = pd.read_csv(f"fact_triplets_processed/{old}_{new}/locality_reord.csv")

    if test_subset:
        # select 100 random probes TODO: Temporary for development
        np.random.seed(42)
        ids = np.random.choice(changed.index, 10, replace=False)
        changed = changed.loc[ids].reset_index(drop=True)
        locality = locality.loc[ids].reset_index(drop=True)

    if not os.path.exists(path + '/edit/edit_unfiltered.csv'):
        print('----------- Edit QA -----------')
        edit_qa = generate_open_qa_pairs(changed, model=model, mode='edit', batch_submission=True)
        save_submission_file(edit_qa, f"fact_qa_sets/{old}_{new}/batch_jobs/{model}/edit")
        submit_batch_request(f"fact_qa_sets/{old}_{new}/batch_jobs/{model}/edit", model=model)

    if not os.path.exists(path + '/loc/loc_unfiltered.csv'):
        print('----------- Locality QA -----------')
        locality_qa = generate_open_qa_pairs(locality, model=model, mode='loc', batch_submission=True)
        save_submission_file(locality_qa, f"fact_qa_sets/{old}_{new}/batch_jobs/{model}/loc")
        submit_batch_request(f"fact_qa_sets/{old}_{new}/batch_jobs/{model}/loc", model=model)
    return


def generate_qa_sets(old, new, model, test_subset=False, pers_model=None, mhop_model=None):
    changed = pd.read_csv(f"fact_triplets_processed/{old}_{new}/changed.csv")
    locality = pd.read_csv(f"fact_triplets_processed/{old}_{new}/locality_reord.csv")
    mhop = pd.read_csv(f"fact_triplets_processed/{old}_{new}/mhop.csv")

    if test_subset:
        # select 100 random probes TODO: Temporary for development
        np.random.seed(42)
        ids = np.random.choice(changed.index, 2, replace=False)
        changed = changed.loc[ids].reset_index(drop=True)
        locality = locality.loc[ids].reset_index(drop=True)
        mhop = mhop.loc[ids].reset_index(drop=True) if mhop is not None else None

    print('----------- Edit QA -----------')
    edit_qa = generate_open_qa_pairs(changed, model=model, mode='edit', batch_submission=False)

    print('----------- Locality QA -----------')
    locality_qa = generate_open_qa_pairs(locality, model=model, mode='loc', batch_submission=False)

    print('----------- Generality QA -----------')
    generality_qa = generate_generality_open_qa_pairs(edit_qa, model=model, batch_submission=False)

    print('----------- Personas QA -----------')
    personas_qa = generate_generality_open_qa_pairs(edit_qa, model=pers_model if pers_model else model, batch_submission=False, character=['mixed'])

    print('----------- Multi-hop QA -----------')
    mhop_qa = generate_mhop_qa_pairs(mhop, model=mhop_model if mhop_model else model, batch_submission=False) if mhop is not None else None

    print('----------- Filtering QA pairs -----------')
    edit_qa, locality_qa, generality_qa, personas_qa = filter_questions(edit_qa, locality_qa, generality_qa, personas_qa)
    mhop_qa = filter_mhop_qa(mhop_qa, edit_qa) if mhop_qa is not None else None

    os.makedirs(f"fact_qa_sets/{old}_{new}/", exist_ok=True)
    edit_qa.to_csv(f"fact_qa_sets/{old}_{new}/edit_{model}.csv", index=False)
    locality_qa.to_csv(f"fact_qa_sets/{old}_{new}/locality_{model}.csv", index=False)
    generality_qa.to_csv(f"fact_qa_sets/{old}_{new}/generality_{model}.csv", index=False)
    personas_qa.to_csv(f"fact_qa_sets/{old}_{new}/personas_{model}.csv", index=False)
    mhop_qa.to_csv(f"fact_qa_sets/{old}_{new}/mhop_{model}.csv", index=False) if mhop_qa is not None else None


def construct_generation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="open", required=True, choices=['open', 'closed'])
    parser.add_argument('--old', type=str, default='20240201')
    parser.add_argument('--new', type=str, default='20240220')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--pers_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--mhop_model', type=str, default='gpt-4o')
    parser.add_argument('--batch', type=bool, default=True)
    parser.add_argument('--test_subset', type=bool, default=False)
    arg = parser.parse_args()
    return arg


if __name__ == "__main__":
    args = construct_generation_args()

    if args.batch:
        done = False
        while not done:
            status = generate_qa_sets_batch(args.old, args.new, args.model,
                                            test_subset=args.test_subset,
                                            pers_model=args.pers_model,
                                            mhop_model=args.mhop_model)
            done = True if status == 'done' else False
    else:
        generate_qa_sets(args.old, args.new, args.model,
                         test_subset=args.test_subset,
                         pers_model=args.pers_model,
                         mhop_model=args.mhop_model)
