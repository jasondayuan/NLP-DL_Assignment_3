from openai import OpenAI
from datasets import load_dataset
import re
import pdb
from tqdm import tqdm
import json

client = OpenAI(api_key="sk-6c9a2b6266414300a01b749da030b011", base_url="https://api.deepseek.com")

def get_response(messages, temp=0):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=temp,
        stream=False
    )
    assert response.choices[0].finish_reason == "stop"
    return response.choices[0].message.content

def extract_answer(answer):
    answer = answer.split('### ')[1]
    answer = re.sub('[^\d\.]', '', answer)
    return answer

def cot(question):
    # Reasoning extraction
    instructions = """You are a helpful assistant that answers math questions.
    The question is delimited by triple backticks. Return the step-by-step
    reasoning and answer for the question. Provide a step-by-step reasoning 
    process leading to the final answer. Ensure that each step is clearly explained 
    and logically follows from the previous one.
    """
    user_content = f"```{question}``` Let's think step by step."
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
    user_content += response
    user_content += "What is the final answer?"
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content}
    ]
    response = get_response(messages, temp=0)

    answer = response.split('Answer: ')[1][:-1]
    answer = re.sub('[^\d\.]', '', answer)

    return answer

def test_cot(cont=False):
    dataset = load_dataset('gsm8k', 'main')  

    if cont:
        with open('results/cot.json', 'r') as f:
            prev_results = json.load(f)
    else:
        prev_results = []
    start_idx = len(prev_results)

    total_samples = 0
    correct_samples = 0
    progress_bar = tqdm(dataset['test'])
    for i, sample in enumerate(progress_bar):
        if i < start_idx:
            total_samples += 1
            correct_samples += prev_results[i]['correct']
            continue

        question = sample['question']
        answer = sample['answer']

        pred_answer = cot(question)
        gt_answer = extract_answer(answer)

        total_samples += 1
        if float(pred_answer) == float(gt_answer):
            correct_samples += 1

        progress_bar.set_postfix(accuracy=correct_samples/total_samples)

        prev_results.append({'question': question, 
                             'pred_answer': pred_answer, 
                             'gt_answer': gt_answer, 
                             'correct': float(pred_answer) == float(gt_answer)})
        
        with open('results/cot.json', 'w') as f:
            json.dump(prev_results, f, indent=4)

    print(f"CoT Accuracy: {correct_samples/total_samples}")

if __name__ == '__main__':
    test_cot(cont=True)