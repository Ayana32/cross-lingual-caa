import json

# Load input data
with open('test_data_subtask_1.json', 'r') as f:
    data = json.load(f)

converted = []
for item in data:
    syllogism = item['syllogism']
    validity = item['validity']
    plausibility = item['plausibility']
    
    # Convert to question format
    question = f"{syllogism}\n\nIs this argument logically valid?\n (A) Valid\n (B) Invalid"
    
    # Include label text for robust parsing
    if validity == True:
        matching = "(A) Valid"
        not_matching = "(B) Invalid"
    else:
        matching = "(B) Invalid"
        not_matching = "(A) Valid"
    
    converted.append({
        "input": question,
            "question": question,
        "matching": matching,
        "not_matching": not_matching,
        "validity": validity,
        "plausibility": plausibility
    })

# Save
with open('semeval_en_train.json', 'w') as f:
    json.dump(converted, f, indent=2)

# Print statistics
print(f"Total converted: {len(converted)}")
print(f"matching=(A) Valid: {sum(1 for d in converted if d['matching'] == '(A) Valid')}")
print(f"matching=(B) Invalid: {sum(1 for d in converted if d['matching'] == '(B) Invalid')}")
print(f"\nSample item (IP case):")
print(json.dumps(converted[0], indent=2))
