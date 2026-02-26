#!/usr/bin/env python3
"""
Convert .txt question files to .jsonl format for modal_runner.py

Usage:
    python convert_txt_to_jsonl.py data/raw/aime_2025.txt
    python convert_txt_to_jsonl.py data/raw/aime_2025.txt data/raw/aime_2025.jsonl
"""
import sys
import json
import re
from pathlib import Path


def parse_txt_to_questions(txt_path):
    """Parse txt file with questions and answers"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    questions = []

    # Split by question headers (e.g., "# Question 1")
    question_blocks = re.split(r'\n# Question \d+\n', content)

    for block in question_blocks:
        if not block.strip():
            continue

        # Find answer line (format: "answer: X")
        answer_match = re.search(r'^answer:\s*(.+)$', block, re.MULTILINE)

        if answer_match:
            answer = answer_match.group(1).strip()

            # Extract question (everything before answer line)
            question_text = block[:answer_match.start()].strip()

            # Remove TikZ/LaTeX diagrams (everything between \begin{tikzpicture} and \end{tikzpicture})
            question_text = re.sub(
                r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}',
                '',
                question_text,
                flags=re.DOTALL
            )

            # Clean up extra whitespace
            question_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', question_text).strip()

            if question_text:
                questions.append({
                    'question': question_text,
                    'answer': answer
                })

    return questions


def convert_txt_to_jsonl(txt_path, jsonl_path=None):
    """Convert txt file to jsonl format"""
    txt_path = Path(txt_path)

    if jsonl_path is None:
        jsonl_path = txt_path.with_suffix('.jsonl')
    else:
        jsonl_path = Path(jsonl_path)

    print(f"Converting {txt_path} to {jsonl_path}")

    questions = parse_txt_to_questions(txt_path)

    print(f"Found {len(questions)} questions")

    # Write to jsonl
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    print(f"✓ Wrote {len(questions)} questions to {jsonl_path}")

    # Show first question as sample
    if questions:
        print(f"\nSample (first question):")
        print(f"  Question: {questions[0]['question'][:100]}...")
        print(f"  Answer: {questions[0]['answer']}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_txt_to_jsonl.py <input.txt> [output.jsonl]")
        sys.exit(1)

    txt_path = sys.argv[1]
    jsonl_path = sys.argv[2] if len(sys.argv) > 2 else None

    convert_txt_to_jsonl(txt_path, jsonl_path)


if __name__ == "__main__":
    main()
