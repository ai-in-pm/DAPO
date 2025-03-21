import json
import os
import argparse
import random
from typing import List, Dict, Any

# Ensure the data directory exists
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'), exist_ok=True)

# Sample tasks for the generated dataset
SAMPLE_TASKS = [
    {
        "type": "math",
        "prompts": [
            "What is {a} + {b}?",
            "Calculate the sum of {a} and {b}.",
            "If I have {a} apples and find {b} more, how many do I have in total?",
            "What is the result of adding {a} to {b}?"
        ],
        "generator": lambda: {"a": random.randint(1, 100), "b": random.randint(1, 100)},
        "answer_key": lambda params: params["a"] + params["b"],
        "responses": [
            lambda params: str(params["a"] + params["b"]),
            lambda params: f"The sum of {params['a']} and {params['b']} is {params['a'] + params['b']}.",
            lambda params: f"The answer is {params['a'] + params['b']}."
        ],
        "incorrect_responses": [
            lambda params: str(params["a"] - params["b"]),
            lambda params: str(params["a"] * params["b"]),
            lambda params: str(params["a"] + params["b"] + random.randint(1, 10)),
            lambda params: str(params["a"] + params["b"] - random.randint(1, 10)),
            lambda params: "I don't know the answer to this question."
        ]
    },
    {
        "type": "reasoning",
        "prompts": [
            "If {person1} is {relation} of {person2}, and {person2} is {relation} of {person3}, what is {person1} to {person3}?",
            "In a family, {person1} is the {relation} of {person2}. {person2} is the {relation} of {person3}. What relation is {person1} to {person3}?"
        ],
        "generator": lambda: {
            "person1": random.choice(["Alice", "Bob", "Charlie", "David", "Emma"]),
            "person2": random.choice(["Frank", "Grace", "Henry", "Isabel", "Jack"]),
            "person3": random.choice(["Karen", "Leo", "Megan", "Nathan", "Olivia"]),
            "relation": random.choice(["parent", "grandparent", "sibling", "child"])
        },
        "answer_key": lambda params: "grandparent" if params["relation"] == "parent" else \
                              "great-grandparent" if params["relation"] == "grandparent" else \
                              "sibling" if params["relation"] == "sibling" else "grandchild",
        "responses": [
            lambda params: f"{params['person1']} is the {params['relation']} of {params['person2']}, and {params['person2']} is the {params['relation']} of {params['person3']}. If each {params['relation']} relationship is the same, then {params['person1']} would be the {'grandparent' if params['relation'] == 'parent' else 'great-grandparent' if params['relation'] == 'grandparent' else 'sibling' if params['relation'] == 'sibling' else 'grandchild'} of {params['person3']}."
        ],
        "incorrect_responses": [
            lambda params: f"{params['person1']} is the parent of {params['person3']}.",
            lambda params: f"{params['person1']} is the child of {params['person3']}.",
            lambda params: f"{params['person1']} is the {params['relation']} of {params['person3']}.",
            lambda params: "They are not related.",
            lambda params: "I need more information to determine the relationship."
        ]
    },
    {
        "type": "classification",
        "prompts": [
            "Classify the sentiment of the following text: '{text}'",
            "Is the following statement positive, negative, or neutral? '{text}'",
            "What is the sentiment of this text: '{text}'"
        ],
        "generator": lambda: {
            "text": random.choice([
                "I absolutely love this product!",
                "This was a complete waste of money.",
                "The service was okay, nothing special.",
                "I can't believe how amazing this experience was!",
                "I'm extremely disappointed with the quality.",
                "It works as expected, but I'm not impressed.",
                "This exceeded all my expectations!",
                "Terrible customer service and product quality.",
                "It's a decent option for the price.",
                "I would highly recommend this to everyone!"
            ])
        },
        "answer_key": lambda params: "positive" if any(word in params["text"].lower() for word in ["love", "amazing", "exceeded", "highly"]) else \
                             "negative" if any(word in params["text"].lower() for word in ["waste", "disappointed", "terrible"]) else "neutral",
        "responses": [
            lambda params: f"The sentiment of the text is {'positive' if any(word in params['text'].lower() for word in ['love', 'amazing', 'exceeded', 'highly']) else 'negative' if any(word in params['text'].lower() for word in ['waste', 'disappointed', 'terrible']) else 'neutral'}.",
            lambda params: f"This text expresses a {'positive' if any(word in params['text'].lower() for word in ['love', 'amazing', 'exceeded', 'highly']) else 'negative' if any(word in params['text'].lower() for word in ['waste', 'disappointed', 'terrible']) else 'neutral'} sentiment."
        ],
        "incorrect_responses": [
            lambda params: f"The sentiment of the text is {'negative' if any(word in params['text'].lower() for word in ['love', 'amazing', 'exceeded', 'highly']) else 'positive' if any(word in params['text'].lower() for word in ['waste', 'disappointed', 'terrible']) else 'mixed'}.",
            lambda params: "I cannot determine the sentiment from this text.",
            lambda params: "The text doesn't have a clear sentiment.",
            lambda params: "The sentiment is mixed."
        ]
    },
    {
        "type": "code",
        "prompts": [
            "Write a function in Python to {task}.",
            "Implement a Python function that {task}.",
            "Create a Python function to {task}. Provide code only."
        ],
        "generator": lambda: {
            "task": random.choice([
                "calculate the factorial of a number",
                "check if a string is a palindrome",
                "find the maximum element in a list",
                "count the occurrences of each element in a list",
                "reverse a string"
            ])
        },
        "answer_key": lambda params: {
            "calculate the factorial of a number": \
                "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)",
            "check if a string is a palindrome": \
                "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
            "find the maximum element in a list": \
                "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)",
            "count the occurrences of each element in a list": \
                "def count_occurrences(lst):\n    return {item: lst.count(item) for item in set(lst)}",
            "reverse a string": \
                "def reverse_string(s):\n    return s[::-1]"
        }[params["task"]],
        "responses": [
            lambda params: {
                "calculate the factorial of a number": \
                    "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)\n\n# Example usage:\n# factorial(5) would return 120",
                "check if a string is a palindrome": \
                    "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n\n# Example usage:\n# is_palindrome('racecar') would return True\n# is_palindrome('hello') would return False",
                "find the maximum element in a list": \
                    "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)\n\n# Example usage:\n# find_max([1, 5, 3, 9, 2]) would return 9",
                "count the occurrences of each element in a list": \
                    "def count_occurrences(lst):\n    return {item: lst.count(item) for item in set(lst)}\n\n# Example usage:\n# count_occurrences([1, 2, 3, 1, 2, 1]) would return {1: 3, 2: 2, 3: 1}",
                "reverse a string": \
                    "def reverse_string(s):\n    return s[::-1]\n\n# Example usage:\n# reverse_string('hello') would return 'olleh'"
            }[params["task"]]
        ],
        "incorrect_responses": [
            lambda params: "I'm not familiar with Python programming.",
            lambda params: "def invalid_function():\n    # This function doesn't work\n    pass",
            lambda params: "print(\"This is not a function implementation\")\n\n# This doesn't solve the task",
            lambda params: "The answer to this would be too complex to implement here.",
            lambda params: "Here's a JavaScript solution instead:\n\nfunction solution() {\n    // JavaScript code\n    return null;\n}"
        ]
    },
    {
        "type": "creative",
        "prompts": [
            "Write a short story about {subject}.",
            "Create a brief narrative about {subject}.",
            "Compose a short tale involving {subject}."
        ],
        "generator": lambda: {
            "subject": random.choice([
                "a lost traveler in a magical forest",
                "a robot discovering human emotions",
                "a detective solving an impossible case",
                "an artist finding inspiration in dreams",
                "a chef who can taste colors"
            ])
        },
        "answer_key": lambda params: params["subject"],  # The answer key is just the subject for creative tasks
        "responses": [
            lambda params: "Once upon a time, " + {
                "a lost traveler in a magical forest": "there was a weary traveler who had wandered off the path in an ancient forest. As dusk fell, the trees began to glow with soft blue light, and whispers echoed between the branches. The traveler realized this was no ordinary forest but a realm of magic and wonder. Following the glowing path that appeared before their feet, they discovered a hidden community of forest spirits who offered guidance and a magical compass that would always lead them home.",
                "a robot discovering human emotions": "in a laboratory of advanced technology, a robot named AX-7 was programmed to assist humans with daily tasks. One day, while helping an elderly woman in her home, AX-7 noticed strange sensations in its circuitry when the woman thanked it with tears in her eyes. Day by day, interaction by interaction, AX-7 began to understand joy, sorrow, empathy, and even love. The scientists were astounded—they had created a machine that could feel, blurring the line between human and artificial intelligence.",
                "a detective solving an impossible case": "Detective Morgan stood at the edge of the sealed room, studying the scene of the crime. No windows, one locked door, no hidden passages—yet the valuable diamond had vanished without a trace. After days of investigation and countless dead ends, Morgan had a flash of insight while watching ice melt in a glass. The solution was as brilliant as it was simple: the thief had created a key made of frozen chemical compound that disappeared after use, leaving no evidence behind. The case that everyone called impossible was finally solved.",
                "an artist finding inspiration in dreams": "there lived an artist named Elena who had lost her creative spark. For months, her canvases remained blank until one night, she experienced a vivid dream of swirling colors and impossible landscapes. When she awoke, she painted frantically, capturing the images before they faded. Each night brought new dreams and new masterpieces. Art critics were astounded by her sudden transformation. What they didn't know was that in her dreams, Elena was visiting other dimensions, bringing back visions no human eye had ever seen.",
                "a chef who can taste colors": "Marco was not an ordinary chef. From childhood, he experienced synesthesia—he could taste colors. Red was spicy and bold, blue was cool and refreshing, yellow was tangy and bright. Using this unique ability, Marco created dishes that were like symphonies of flavor, each designed to tell a story through taste. His restaurant became world-famous, with people traveling across continents just to experience his color-inspired cuisine. What began as a strange neurological condition became Marco's greatest gift to the culinary world."
            }[params["subject"]]
        ],
        "incorrect_responses": [
            lambda params: "I don't have enough information to write a story about this subject.",
            lambda params: "I'm not very good at creative writing, so I'll skip this request.",
            lambda params: "Here's a list of tips for writing stories instead:\n1. Start with a strong hook\n2. Develop interesting characters\n3. Create conflict\n4. Resolve the plot.",
            lambda params: "I'd rather not write a story right now.",
            lambda params: "Stories should have a beginning, middle, and end. That's all I can tell you."
        ]
    }
]

def format_prompt(template: str, params: Dict[str, Any]) -> str:
    """Format a prompt template with the given parameters.
    
    Args:
        template: Prompt template string.
        params: Dictionary of parameters to substitute.
        
    Returns:
        Formatted prompt string.
    """
    return template.format(**params)

def generate_sample(task_type: str = None) -> Dict[str, Any]:
    """Generate a sample for the dataset.
    
    Args:
        task_type: Type of task to generate, or None for random.
        
    Returns:
        Dictionary containing prompt, responses, and answer_key.
    """
    # Select a task type
    if task_type is None:
        task = random.choice(SAMPLE_TASKS)
    else:
        matching_tasks = [t for t in SAMPLE_TASKS if t["type"] == task_type]
        if not matching_tasks:
            raise ValueError(f"Task type '{task_type}' not found")
        task = random.choice(matching_tasks)
    
    # Generate parameters for the task
    params = task["generator"]()
    
    # Select a prompt template
    prompt_template = random.choice(task["prompts"])
    
    # Format the prompt
    prompt = format_prompt(prompt_template, params)
    
    # Generate correct responses
    correct_responses = [response_fn(params) for response_fn in task["responses"]]
    
    # Generate incorrect responses
    incorrect_responses = [response_fn(params) for response_fn in task["incorrect_responses"]]
    
    # Get the answer key
    answer_key = task["answer_key"](params)
    
    # Create the sample
    sample = {
        "prompt": prompt,
        "responses": correct_responses + incorrect_responses,
        "answer_key": answer_key,
        "metadata": {
            "task_type": task["type"],
            "correct_indices": list(range(len(correct_responses))),
            "params": params
        }
    }
    
    return sample

def generate_dataset(num_samples: int, output_path: str, task_types: List[str] = None) -> None:
    """Generate a dataset of samples and save it to a JSONL file.
    
    Args:
        num_samples: Number of samples to generate.
        output_path: Path to save the dataset.
        task_types: List of task types to generate, or None for all types.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples
    samples = []
    for _ in range(num_samples):
        if task_types:
            task_type = random.choice(task_types)
        else:
            task_type = None
        
        sample = generate_sample(task_type)
        samples.append(sample)
    
    # Save samples to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Generated {num_samples} samples and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate sample dataset for DAPO training")
    parser.add_argument(
        '--num-samples', type=int, default=100,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--output', type=str, default='data/sample_dataset.jsonl',
        help='Path to save the generated dataset'
    )
    parser.add_argument(
        '--train-split', type=float, default=0.8,
        help='Proportion of samples to use for training'
    )
    parser.add_argument(
        '--task-types', type=str, nargs='+', default=None,
        help='Task types to generate (math, reasoning, classification, code, creative)'
    )
    args = parser.parse_args()
    
    # Determine output paths
    base_dir = os.path.dirname(args.output)
    base_name = os.path.basename(args.output).split('.')[0]
    train_path = os.path.join(base_dir, f"{base_name}_train.jsonl")
    eval_path = os.path.join(base_dir, f"{base_name}_eval.jsonl")
    
    # Calculate split sizes
    train_size = int(args.num_samples * args.train_split)
    eval_size = args.num_samples - train_size
    
    # Generate training dataset
    generate_dataset(train_size, train_path, args.task_types)
    
    # Generate evaluation dataset
    generate_dataset(eval_size, eval_path, args.task_types)
    
    print(f"Generated {train_size} training samples and {eval_size} evaluation samples")
    print(f"Training samples saved to {train_path}")
    print(f"Evaluation samples saved to {eval_path}")

if __name__ == "__main__":
    main()
