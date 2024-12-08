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

def naive_prompting(question):
    instructions = """You are a helpful assistant that answers math questions.
    The question is delimited by triple backticks. Return the final answer, 
    prefixed by 'Answer: ', ended with '.'.
    """
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"```{question}```"}
    ]

    response = get_response(messages, temp=0)

    answer = response.split('Answer: ')[1][:-1]
    answer = re.sub('[^\d\.]', '', answer)

    return answer

def test_naive_prompting(cont=False):
    dataset = load_dataset('gsm8k', 'main')  

    if cont:
        with open('results/naive.json', 'r') as f:
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

        pred_answer = naive_prompting(question)
        gt_answer = extract_answer(answer)

        total_samples += 1
        correct = 0
        try:
            if float(pred_answer) == float(gt_answer):
                correct_samples += 1
                correct = 1
        except ValueError:
            print(f"Pred: {pred_answer}, GT: {gt_answer}")

        progress_bar.set_postfix(accuracy=correct_samples/total_samples)

        prev_results.append({'question': question, 
                             'pred_answer': pred_answer, 
                             'gt_answer': gt_answer, 
                             'correct': correct})
        
        with open('results/naive.json', 'w') as f:
            json.dump(prev_results, f, indent=4)

    print(f"Naive Prompting Accuracy: {correct_samples/total_samples}")

if __name__ == '__main__':
    test_naive_prompting(cont=True)