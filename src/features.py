import numpy as np
import pandas as pd
from tqdm import tqdm


def get_data(user_sample, win_code, test_set=False):
    last_activity = 0
    user_activities_count = {'Clip': 0,
                             'Activity': 0, 'Assessment': 0, 'Game': 0}
    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    durations = []
    for i, session in user_sample.groupby('game_session', sort=False):
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        if test_set is True:
            second_condition = True
        else:
            if len(session) > 1:
                second_condition = True
            else:
                second_condition = False

        if (session_type == 'Assessment') & (second_condition):
            all_attempts = session.query(
                f'event_code == {win_code[session_title]}')
            true_attempts = all_attempts['event_data'].str.contains(
                'true').sum()
            false_attempts = all_attempts['event_data'].str.contains(
                'false').sum()
            features = user_activities_count.copy()
            features['session_title'] = session['title'].iloc[0]
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts
            accumulated_uncorrect_attempts += false_attempts
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append(
                (session.iloc[-1, 2] - session.iloc[0, 2]).seconds)
            features['accumulated_accuracy'] = accumulated_accuracy / \
                counter if counter > 0 else 0
            accuracy = true_attempts / \
                (true_attempts+false_attempts) if (true_attempts +
                                                   false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1

            features.update(accuracy_groups)
            features['accumulated_accuracy_group'] = accumulated_accuracy_group / \
                counter if counter > 0 else 0
            features['accumulated_actions'] = accumulated_actions
            accumulated_accuracy_group += features['accuracy_group']
            accuracy_groups[features['accuracy_group']] += 1
            if test_set is True:
                all_assessments.append(features)
            else:
                if true_attempts+false_attempts > 0:
                    all_assessments.append(features)

            counter += 1

        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type

    if test_set:
        return [all_assessments[-1]]
    return all_assessments


def generate_features(df, win_code, mode):
    if mode == 'train':
        total = 17000
        test_set = False
    elif mode == 'test':
        total = 1000
        test_set = True
    else:
        raise ValueError('mode must be train or test.')
    compiled_data = []
    for i, (ins_id, user_sample) in tqdm(enumerate(df.groupby('installation_id', sort=False)), total=total):
        compiled_data += get_data(user_sample, win_code, test_set)
    return pd.DataFrame(compiled_data)
