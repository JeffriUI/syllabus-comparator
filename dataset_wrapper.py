from pandas import read_csv, DataFrame, concat
from datasets import Dataset
import yaml

def wrap_dataset(targets, topics):
    data = DataFrame()
    
    for topic in topics:
        courses = read_csv(f'scraper/{topic}/courses_{topic}.csv')
        
        # Remove all other columns from scrapping results
        # except 'translated_skills_youll_gain',
        # then rename the column simply to 'course'
        removed_columns = courses.columns.to_list()
        removed_columns.remove('translated_skills_youll_gain')
        courses = courses.drop(columns=(removed_columns))
        courses = courses.dropna(ignore_index=True)
        courses = courses.rename(columns={'translated_skills_youll_gain': 'course'})
        
        # Add a new column in 'course_topic'
        courses['course_topic'] = [topic] * len(courses.index)
        # Copy the current data as template to concatenate later
        df = courses.copy()
        # Add two new columns of the target syllabus and its topic
        target = targets[targets['topic']==topic]['target'].to_numpy()[0]
        courses = courses.join(DataFrame({
            'target': [target] * len(courses.index),
            'target_topic': [topic] * len(courses.index)
        }))

        # Iterate through all the topics available,
        # except the current topic in the outer loop,
        # to concatenate the rest of the course-target pairings
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

        # Apply the labelling according to matching topics
        courses['label'] = [int(l) for l in (courses['course_topic'] == courses['target_topic'])]
        
        # Drop the 'topic' columns for the final returned data,
        # then concatenate to the current compiled data
        courses = courses.drop(columns=['course_topic', 'target_topic'])
        data = concat([data, courses], ignore_index=True)

    # Remove duplicated pairings, resolving labeling conflicts by 
    # prioritizing '1' or 'equivalent' labeling
    # Sort the data by the 'label' columnn in descending order
    data = data.sort_values(by=['label'], ascending=False)
    duplicates = data.duplicated(subset=['course', 'target'], keep='first')
    print(f"Number of duplicate entries by course-target pairings removed: {len(duplicates)}")
    print("The entries with label '1' will take precedence and be kept.")
    data = data.drop_duplicates(subset=['course', 'target'], keep='first')
    
    # Resort and reset indexing to retain original orders without 'jumping' indexes
    data = data.sort_index()
    data = data.reset_index(drop=True)
    print(f"Total number of rows retained in dataset: {len(data.index)}")
    
    return data

def split_dataset(data, dataset_dir, split_test):
    dataset = Dataset.from_pandas(data, preserve_index=False)

    if split_test:
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        # Test split: 20%
        dataset['test'].to_csv(f"{dataset_dir}/test.csv")

        dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
        # Validation split: 80% * 20% = 16%
        dataset['test'].to_csv(f"{dataset_dir}/validation.csv")
        # Train split: 80% * 80% = 64%
        dataset['train'].to_csv(f"{dataset_dir}/train.csv")
    else: 
        dataset = dataset.train_test_split(train_size=0.8, seed=42)
        # Train split: 80%
        dataset['train'].to_csv(f"{dataset_dir}/train.csv")
        # Validation split: 20%
        dataset['test'].to_csv(f"{dataset_dir}/validation.csv")

if __name__ == "__main__":
    targets_path = "datasets/targets.csv"
    config_path = "wrapper_config.yaml"
    
    targets = read_csv(targets_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    for k in config.keys():
        dataset_dir = config[k]['directory']
        topics = config[k]['topics']
        # splits = config[k]['splits']
        split_test = config[k]['split_test']
        data = wrap_dataset(targets, topics=topics)
        split_dataset(data, dataset_dir=dataset_dir, split_test=split_test)