from pathlib import Path

import numpy as np
import pandas as pd
from coolname import generate_slug

from colse.datasets.dataset import Dataset

rng = np.random.default_rng(12345)


def get_first_and_last_name():
    full_name = generate_slug(2)
    first_name, last_name = full_name.split('-')
    """make first letter capital"""
    first_name = first_name.capitalize()
    last_name = last_name.capitalize()
    return first_name, last_name


def generate_one_record():
    """
    Database columns
    1. ID
    2. First name
    3. Last name
    4. Age
    5. Salary
    6. Position
    Returns:

    """
    first_name, last_name = get_first_and_last_name()

    """Age should be within 24 - 70 - normal distribution """
    mean = 35
    std_dev = 15
    age = 0
    # Make sure the number is within the desired range
    while not (20 <= age <= 70):
        age = int(rng.normal(mean, std_dev))

    """gender """
    gender = rng.choice(['M', 'F'], 1, p=[0.6, 0.4])[0]

    gender_based_salary = False
    """Salary should be within 20,000 - 200,000 - normal distribution """
    mean = 80000
    if gender_based_salary:
        mean += 20000 if gender == 'M' else 0
    mean += 60000 * (age - 20) / 50
    std_dev = 1000
    random_number = int(rng.normal(mean, std_dev))
    # Make sure the number is within the desired range
    salary = max(20000, min(random_number, 200000))
    """round to nearest 1000"""
    salary = round(salary, -3)

    """position """
    position = \
        rng.choice(['Software Engineer', 'Data Scientist', 'Product Manager', 'Data Engineer', 'DevOps Engineer'], 1)[0]

    print(
        f'First name: {first_name}, last name: {last_name} Age: {age} Gender : {gender} Salary: {salary} Position : {position}')
    """first_name, last_name, gender, age, salary, position"""
    return {
        'first_name': first_name,
        'last_name': last_name,
        'gender': gender,
        'age': age,
        'salary': salary,
        'position': position
    }


def create_dataset(dataset_name):
    list_of_records = []
    for index in range(100000):
        print(f"Record {index + 1}")
        details = generate_one_record()
        details.update({'ID': index + 1})
        list_of_records.append(details)

    df = pd.DataFrame(list_of_records)
    df.to_csv(dataset_name, index=False)


def get_dataset(dataset_name='employee_dataset.csv'):
    db_path = Path(dataset_name)
    if not db_path.exists():
        print("Database not exists!")
        create_dataset(dataset_name)
    df = pd.read_csv(db_path, index_col=None)
    return Dataset(
        data=df,
        name=dataset_name,
        continuous_columns=['age', 'salary'],
        categorical_columns=['gender', 'position']
    )


if __name__ == '__main__':
    create_dataset(dataset_name="employee_dataset_huge.csv")