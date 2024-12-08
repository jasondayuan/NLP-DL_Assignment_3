from openai import OpenAI
from datasets import load_dataset
import re
import pdb
from tqdm import tqdm
import json

client = OpenAI(api_key="sk-d039ae183d4343c1ba607a5de1a24e63", base_url="https://api.deepseek.com")

def get_response(messages, temp=0):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=temp,
        stream=False
    )
    assert response.choices[0].finish_reason == "stop"
    return response.choices[0].message.content

def cot(question, reflections):
    # Reasoning extraction
    instructions = """You are a helpful assistant that answers math questions. 
    The question is delimited by triple backticks. Return the step-by-step
    reasoning and answer for the question. Provide a step-by-step reasoning 
    process leading to the final answer. Ensure that each step is clearly explained 
    and logically follows from the previous one. You will also be given reflections 
    on your previous unsuccessful reasoning processes, for reference.
    """
    user_content = f"Reflections: {reflections}\n```Question: {question}``` Let's think step by step."
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content}
    ]
    response = get_response(messages, temp=0)

    # Answer extraction
    instructions = """You are a helpful assistant that answers math questions.
    You will be given a math question and its step-by-step reasoning process leading
    to the final answer. Return the final answer, prefixed by 'Answer: ', ended with '.'.
    """
    user_content = f"```Question: {question}``` Let's think step by step.\nAnswer: {response}"
    results = user_content
    user_content += "\nWhat is the final answer?"
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content}
    ]
    response = get_response(messages, temp=0)

    answer = response.split('Answer: ')[1][:-1]
    answer = re.sub('[^\d\.]', '', answer)

    return answer, results

def get_reflection(unsuccessful_example):
    instructions = """You are an advanced reasoning agent that can improve based on self refection. 
    You will be given a previous reasoning trial in which you were given access to relevant context 
    and a question to answer. You were unsuccessful in answering the question either because you guessed 
    the wrong answer  or there is a phrasing discrepancy with your provided answer and the answer key. 
    In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, 
    concise, high level plan that aims to mitigate the same failure. Use complete sentences.
    """
    user_content = f"{unsuccessful_example}"
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content}
    ]
    response = get_response(messages, temp=0)
    return response

# Note: Reflexion stops as soon as the answer is correct, and it focuses on one question
class Reflexion:

    def __init__(self, question, gt_answer):
        self.question = question
        self.gt_answer = gt_answer
        self.unsuccessful_examples = []
        self.reflection_str = ''
        self.succeed = False
        self.n_attempt = 0
        self.pred_answer = None
    
    def run(self):
        if self.succeed:
            return
        if self.n_attempt > 0 and not self.succeed:
            self.reflect()
        self.step()
        self.n_attempt += 1

    def step(self):
        answer, results = cot(self.question, self.reflection_str)
        self.succeed = self.is_correct(answer)
        self.pred_answer = answer
        if not self.succeed:
            self.unsuccessful_examples.append(results + "\nAnswer is INCORRECT.")

    def reflect(self):
        self.reflection_str += "\n" + get_reflection(self.unsuccessful_examples[-1])

    def is_correct(self, answer):
        try:
            gt_answer_numeric = float(self.gt_answer)
            answer_numeric = float(answer)
            return gt_answer_numeric == answer_numeric
        except ValueError:
            return False
    
    def to_dict(self):
        return {
            'question': self.question,
            'ground_truth_answer': self.gt_answer,
            'predicted_answer': self.pred_answer,
            'succeed': self.succeed,
            'unsuccessful_examples': self.unsuccessful_examples,
            'reflection': self.reflection_str
        }

def extract_answer(answer):
    answer = answer.split('### ')[1]
    answer = re.sub('[^\d\.]', '', answer)
    return answer

def main(cont=False):
    dataset = load_dataset('gsm8k', 'main')  

    if cont:
        with open('results/reflexion.json', 'r') as f:
            prev_results = json.load(f)
    else:
        prev_results = []
    start_idx = len(prev_results)

    total_samples = 0
    correct_samples = 0
    max_attempts = 2
    progress_bar = tqdm(dataset['test'])
    for i, sample in enumerate(progress_bar):
        if i < start_idx:
            total_samples += 1
            correct_samples += prev_results[i]['succeed']
            continue

        agent = Reflexion(sample['question'], extract_answer(sample['answer']))

        for _ in range(max_attempts):
            agent.run()
        total_samples += 1
        if agent.succeed:
            correct_samples += 1

        progress_bar.set_postfix(accuracy=correct_samples/total_samples)

        prev_results.append(agent.to_dict())
        
        with open('results/reflexion.json', 'w') as f:
            json.dump(prev_results, f, indent=4)

    print(f"Reflexion Accuracy: {correct_samples/total_samples}")

if __name__ == '__main__':
    main(cont=True)