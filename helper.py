import numpy as np
import pandas as pd
from ast import literal_eval
import math
from functools import partial
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm


def make_feature_data(course):
    """
    Given a course name, we return a dataframe with the student's pre-enrolled 
    courses, semester, and whether the course is dropped.
    """

    # Get pre dataset
    sp22_pre_df = pd.read_csv('./data_by_student/sp22_pre.csv')
    fa22_pre_df = pd.read_csv('./data_by_student/fa22_pre.csv')
    sp23_pre_df = pd.read_csv('./data_by_student/sp23_pre.csv')

    pre_df = pd.concat([sp22_pre_df, fa22_pre_df, sp23_pre_df], axis=0)
    pre_df['exams'] = pre_df['exams'].apply(literal_eval)
    pre_df = pre_df[pre_df['exams'].apply(lambda x: course in x)]
    print(f"{len(pre_df)} students pre-enrolled in {course}")


    # Get post dataset
    sp22_post_df = pd.read_csv('./data_by_student/sp22_post.csv')
    fa22_post_df = pd.read_csv('./data_by_student/fa22_post.csv')
    sp23_post_df = pd.read_csv('./data_by_student/sp23_post.csv')
    post_df = pd.concat([sp22_post_df, fa22_post_df, sp23_post_df], axis=0)
    post_df = post_df.rename(columns=lambda x: x + '-post')
    post_df['dropped'] = post_df['exams-post'].apply(lambda x : not course in x)

    merge = pd.merge(pre_df, post_df, left_on=['student', 'semester'], \
                     right_on=['student-post', 'semester-post'], how='left')
    print(f"{np.sum(merge['dropped'].isna())} data point is nan during processing")
    merge = merge[~merge['dropped'].isna()]

    # Sanity check that course is indeed dropped
    for idx, row in merge.iterrows():
        if row['dropped']:
            assert course in row['exams'] and (not course in row['exams-post'])
        else:
            assert course in row['exams'] and (course in row['exams-post'])
    
    print('Sanity check passed')
    merge = merge.drop(['semester-post', 'student-post', 'exams-post'], axis=1)
    merge['dropped'] = merge['dropped'].replace({True: 1, False: 0})
    print(f'{np.sum(merge["dropped"])} students dropped')


    return merge


def add_course_history(df):
    pass #TODO

def feature_engineer(course):
    df = make_feature_data(course)
    def num_eng_courses(course_list):
        eng_list = ['CS', 'INFO', 'ORIE', 'CEE', 'CHEME', 'BME', 'BEE', 'ECE', 'MAE', 'MSE', 
                    'AEP', 'ENGRC', 'ENGRD', 'ENGRI', 'ENGRG', 'ENMGT', 'SYSEN', 'STSCI', 
                    'MATH', 'PHYS', 'EAS', 'CHEM']
        eng_courses = [1 for course in course_list if course.split('-')[0] in eng_list]
        return sum(eng_courses)
    
    def same_dept_courses(course_list):
        course_dept = course.split('-')[0]
        num_courses = [1 for course in course_list if course.split('-')[0] == course_dept]
        return sum(num_courses)
    
    def sum_course_levels(course_list):
        '''
        Define a function to calculate the sum of course levels for a given list of courses
        '''
        course_levels = [math.floor(int(course.split('-')[1]) / 1000) for course in course_list]
        return sum(course_levels)
    
    def get_year(value):
        return int('20' + value[2:])
    
    def get_sem(value):
        if value[:2] == 'sp':
            return 'spring'
        else:
            return 'fall'
    
    def cs_by_level(exams, level):
        # e.g. level = 3, then return number of cs3000+ courses
        counter = 0
        for i in exams:
            if i.startswith('CS-' + str(level)):
                counter += 1
        return counter

    df2 = df.copy()
    df2['num_course'] = df2['exams'].apply(len)
    df2['num_eng_course'] = df2['exams'].apply(num_eng_courses)
    df2['num_same_dept_course'] = df2['exams'].apply(same_dept_courses)
    df2['course_level_sum'] = df2['exams'].apply(sum_course_levels)
    df2['year'] = df2['semester'].apply(get_year)
    df2['sem'] = df2['semester'].apply(get_sem)
    dummy_cols = pd.get_dummies(df2['sem'], prefix='sem')
    df2 = pd.concat([df2, dummy_cols], axis=1)

    df2 = df2.drop(columns=['sem', 'exams', 'student'], axis=1)


    # # CS by course level
    # df2['num_cs7000'] = df2['exams'].apply(partial(cs_by_level,level=7))
    # df2['num_cs6000'] = df2['exams'].apply(partial(cs_by_level,level=6))
    # df2['num_cs5000'] = df2['exams'].apply(partial(cs_by_level,level=5))
    # df2['num_cs4000'] = df2['exams'].apply(partial(cs_by_level,level=4))
    # df2['num_cs3000'] = df2['exams'].apply(partial(cs_by_level,level=3))
    # df2['num_cs2000'] = df2['exams'].apply(partial(cs_by_level,level=2))
    # df2['num_cs1000'] = df2['exams'].apply(partial(cs_by_level,level=1))

    return df2
    

def make_semester_specific_train_test(all_data_df, target_sem, past_sems, course_history = False):
    '''
    Use data in past_sems as train data and data in target_sem as test data.
    '''
    merge = all_data_df

    if course_history: # If course_history is required, provide the past enrollment data for each student.
        pass

    train_data_df = merge[merge['semester'].isin(past_sems)].drop(columns=['semester'], axis=1)
    test_data_df = merge[merge['semester'] == target_sem].drop(columns=['semester'], axis=1)

    num_drop_train = len(train_data_df[train_data_df['dropped'] == 1])
    assert num_drop_train != 0, "Error: Number of dropped students in training data should not be zero"
    print(f"{num_drop_train} students dropped in the train data")
 
    target_column_name = "dropped"
    # separate the feature and target variables for the train and test sets
    X_train = train_data_df.drop(columns=[target_column_name]).to_numpy()
    y_train = train_data_df[target_column_name].to_numpy()
    X_test = test_data_df.drop(columns=[target_column_name]).to_numpy()
    y_test = test_data_df[target_column_name].to_numpy()



    return X_train, y_train, X_test, y_test


def make_random_train_test(all_data_df, train_ratio, random_state = 2):
    '''
    random train test split.
    '''
    all_data_df = all_data_df.drop('semester', axis=1)
    # shuffle the data randomly
    shuffled_data_df = all_data_df.sample(frac=1, random_state = random_state)

    # calculate the index to split the data into train and test sets
    split_index = int(train_ratio * len(shuffled_data_df))

    # split the shuffled data into train and test sets
    train_data_df = shuffled_data_df[:split_index]
    test_data_df = shuffled_data_df[split_index:]

    target_column_name = "dropped"
    # separate the feature and target variables for the train and test sets
    X_train = train_data_df.drop(columns=[target_column_name]).to_numpy()
    y_train = train_data_df[target_column_name].to_numpy()
    X_test = test_data_df.drop(columns=[target_column_name]).to_numpy()
    y_test = test_data_df[target_column_name].to_numpy()

    return X_train, y_train, X_test, y_test



# Define a custom scoring function
def average_difference(model, x_test, y_test):
    predicted_dropouts = np.sum(model.predict_proba(x_test)[:,1])
    # Get the actual number of dropouts for the course
    actual_dropouts = y_test.sum()
    # Calculate the squared difference between predicted and actual dropouts
    difference = (predicted_dropouts - actual_dropouts)**2
    # Return the negative difference as the score (lower difference is better)
    return -difference

def param_tuning(classifier, param_grid, courses):
    """
    Perform parameter tuning on a classifier for each course, the score of which is based on the squared semester-specific prediction difference.
    """
    # Create an empty dictionary to store the best models for each course
    best_models = {}
    
    # Loop over the unique courses in the dataframe
    for course in tqdm(courses):
        course_df = feature_engineer(course)
        x_train, y_train, x_test, y_test = make_semester_specific_train_test(course_df, target_sem = "sp23",
                                                                            past_sems = ["fa22", "sp22"])
        X = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        # Create an instance of the GridSearchCV class
        shufflesplit = ShuffleSplit(n_splits=5, train_size=0.8, test_size=0.2)        
        grid_search = GridSearchCV(classifier(random_state=42), param_grid, scoring=average_difference, cv=shufflesplit)
        # Fit the grid search to the data
        grid_search.fit(X, y)
        # Print the best parameters and score for the classifier for the course
        print(course)
        print(grid_search.best_params_)
        print(grid_search.best_score_)
        # Store the best estimator in the dictionary
        best_models[course] = grid_search.best_estimator_
    
    # Return the dictionary of best models
    return best_models
