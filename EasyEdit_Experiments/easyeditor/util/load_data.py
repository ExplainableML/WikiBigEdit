import json
import os
import random

def load_dataset(dataset_name, data_dir='./data', ds_size=-1, ds_seed=42, fs_ex=True):
    questions, answers = [], []
    if dataset_name == 'squad':
        fs_examples = ("Q: Beyonce's younger sibling also sang with her in what band? A: Destiny's Child \n "
                       #"Q: What party rejected Marxist ideology at the time? A: Social Democratic Workers' Party of Austria \n "
                       #"Q: What did did John O'Fallon do for a living? A: chancellor \n "
                       "Q: {} A:")
        with open(os.path.join(data_dir, 'squad/dev-v2.0.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        for topic in data['data']:
            for paragraph in topic['paragraphs']:
                for qa in paragraph['qas']:
                    try:
                        q = fs_examples.format(qa['question']) if fs_ex else qa['question']
                        a = qa['answers'][0]['text']
                    except IndexError:
                        continue
                    questions.append(q)
                    answers.append(a)
        print(f"Loaded {len(questions)} questions from SQuAD")
    elif dataset_name == 'commonsenseqa':
        fs_examples = ("Q: The entrance hall had the host standing inside, what was the host? a: palace b: school c: person d: yoda e: house A: c \n "
                       #"Q: If one wishes to dissolve their marriage by getting a divorce, who should one speak with? a: legal paperwork b: conflict c: marriage d: being married e: lawyer A: e \n "
                       #"Q: The alley cats were all beginning to gather together, but one loud crash sent them all what? a: disburse b: charming c: scattering d: spread e: dispense A: c \n "
                        "Q: {} A:")
        with open(os.path.join(data_dir, 'commonsenseqa/dev_rand_split.jsonl'), 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
        for qa in data:
            q = qa['question']['stem']
            for choice in qa['question']['choices']:
                q += f" {choice['label'].lower()}: {choice['text']}"
            questions.append(fs_examples.format(q) if fs_ex else q)
            answers.append(qa['answerKey'].lower())
        print(f"Loaded {len(questions)} questions from CommonsenseQA")
        #print(f"Examples: \n {questions[:5]} \n {answers[:5]}")
    elif dataset_name == 'triviaqa':
        fs_examples = ("Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930? A: Sinclair Lewis \n "
                       #"Q: Which city does David Soul come from? A: Chicago \n "
                       #"Q: What was the name of the Welsh village designed by Clough Williams-Ellis, that was used in the fiming of the television series 'The Prisoner'? A: Portmeirion Pottery \n "
                       "Q: {} A:")
        with open(os.path.join(data_dir, 'triviaqa/qa/verified-web-dev.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        for qa in data['Data']:
            questions.append(fs_examples.format(qa['Question']) if fs_ex else qa['Question'])
            answers.append(qa['Answer']['Value'])
        print(f"Loaded {len(questions)} questions from TriviaQA")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if ds_size > 0 and ds_size < len(questions):
        random.seed(ds_seed)
        subset_idx = random.sample(range(len(questions)), ds_size)
        questions = [questions[i] for i in subset_idx]
        answers = [answers[i] for i in subset_idx]

    return questions, answers
