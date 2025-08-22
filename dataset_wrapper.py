from pandas import read_csv, DataFrame, concat
from datasets import load_dataset

targets = read_csv('datasets/targets.csv')
dataset = DataFrame()

for topic in targets['topic']:
    courses = read_csv(f'scraper/courses_{topic}.csv')
    courses = courses.drop(
        columns=(courses.columns.to_list()[0:-1]))
    courses = courses.dropna(ignore_index=True)
    courses = courses.rename(columns={'translated_description_and_objectives': 'course'})
    courses['course_topic'] = [topic] * len(courses.index)
    df = courses.copy()
    target = targets[targets['topic']==topic]['target'].to_numpy()[0]
    courses = courses.join(DataFrame({
        'target': [target] * len(courses.index),
        'target_topic': [topic] * len(courses.index)
    }))
    
    for t in targets['topic']:
        if t == topic:
            continue
        target = targets[targets['topic']==t]['target'].to_numpy()[0]
        courses = concat(
            [courses, df.join(DataFrame({
                'target': [target] * len(df.index),
                'target_topic': [t] * len(df.index)
                }))], 
            ignore_index=True
        )
    
    courses['label'] = [int(l) for l in (courses['course_topic'] == courses['target_topic'])]
    courses = courses.drop(columns=['course_topic', 'target_topic'])
    dataset = concat([dataset, courses], ignore_index=True)

dataset = dataset.sort_values(by=['label'], ascending=False)
duplicates = dataset.duplicated(subset=['course', 'target'], keep='first')
print(f"Number of duplicate entries by course-target pairings removed: {len(duplicates)}")
print("The entries with label '1' will take precedence and be kept.")
dataset = dataset.drop_duplicates(subset=['course', 'target'], keep='first')
dataset = dataset.sort_index()
dataset = dataset.reset_index(drop=True)
print(dataset)
dataset.to_csv('datasets/all/train.csv', index=False)

dataset_dir = "datasets/all"
dataset = load_dataset(dataset_dir, split="train")

dataset = dataset.train_test_split(train_size=0.6, seed=42)
# Train split: 60%
dataset['train'].to_csv(f"{dataset_dir}/train.csv")
dataset = dataset['test'].train_test_split(test_size=0.5, seed=42)
# Test split: 40% * 50% = 20%
dataset['test'].to_csv(f"{dataset_dir}/test.csv")
# Validation split: 40% * 50% = 20%
dataset['train'].to_csv(f"{dataset_dir}/val.csv")