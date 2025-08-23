from pandas import read_csv, DataFrame, concat
from datasets import Dataset

targets = read_csv('datasets/targets.csv')
dataset = DataFrame()

for topic in targets['topic']:
    courses = read_csv(f'scraper/{topic}/courses_{topic}.csv')
    removed_columns = courses.columns.to_list()
    removed_columns.remove('translated_skills_youll_gain')
    courses = courses.drop(columns=(removed_columns))
    courses = courses.dropna(ignore_index=True)
    courses = courses.rename(columns={'translated_skills_youll_gain': 'course'})
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
dataset = Dataset.from_pandas(dataset, preserve_index=False)

dataset_dir = "datasets/all"

dataset = dataset.train_test_split(test_size=0.2, seed=42)
# Test split: 20%
dataset['test'].to_csv(f"{dataset_dir}/test.csv")
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
# Validation split: 80% * 20% = 16%
dataset['test'].to_csv(f"{dataset_dir}/validation.csv")
# Train split: 80% * 80% = 64%
dataset['train'].to_csv(f"{dataset_dir}/train.csv")