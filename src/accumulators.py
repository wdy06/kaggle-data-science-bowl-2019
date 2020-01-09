from collections import defaultdict

import numpy as np

import utils


class UserStatsAcc:
    def __init__(self, win_code, is_test):
        super().__init__()
        self.assessment_acc = AssessmentAcc(win_code)
        self.session_acc = SessionAcc()
        self.win_code = win_code
        self.user_activities_count = defaultdict(lambda: defaultdict(int))
        self.last_session_id = defaultdict(lambda: None)
        self.last_activity = defaultdict(int)
        self.last_title = defaultdict(lambda: None)
        self.last_world = {}
        self.last_time = {}
        self.durations = defaultdict(list)
        self.is_test = is_test

    def update_acc(self, row):
        ins_id = row['installation_id']
        session_id = row['game_session']
        session_type = row['type']
        if session_id != self.last_session_id.get(ins_id):
            self.user_activities_count[ins_id][session_type] += 1
        self.user_activities_count[ins_id]['all_actions_count'] += 1
        self.last_session_id[ins_id] = session_id
        self.last_activity[ins_id] = session_type
        self.last_title[ins_id] = row['title']
        self.last_world[ins_id] = row['world']
        self.last_time[ins_id] = row['timestamp']
        if row['end_of_game']:
            self.user_activities_count[ins_id]['game_count'] += 1
        # update chile accumulators
        self.assessment_acc.update_acc(row)
        self.session_acc.update_acc(row)
        ass_stats = self.assessment_acc.get_stats(row)
        session_stats = self.session_acc.get_stats(row)
        # self.is_evaluate_timing = (session_type == 'Assessment') and (row['end_of_game']) \
        #     and (ass_stats['step_count_in_game'] > 1)
        # print(ins_id, row['game_session'], session_type, row['end_of_game'],
        #       session_stats['step_count_in_game'], self.is_evaluate_timing)
        if self.is_evaluate_timing(row, ass_stats):
            self.user_activities_count[ins_id]['evaluate_count'] += 1
            self.user_activities_count[ins_id]['true_attempts_count'] += ass_stats['true_attempts_count']
            self.user_activities_count[ins_id]['false_attempts_count'] += ass_stats['false_attempts_count']
            self.user_activities_count[ins_id]['accumulated_accracy'] += ass_stats['accuracy']
            self.user_activities_count[ins_id]['accumulated_accracy_group'] += ass_stats['accuracy_group']
            accuracy_group = ass_stats['accuracy_group']
            self.user_activities_count[ins_id][f'acc_group_{accuracy_group}_count'] += 1
            self.durations[ins_id].append(session_stats['session_time_length'])

    def get_stats(self, row):
        ins_id = row['installation_id']
        output = {}
        output['session_title'] = row['title']
        output['game_count'] = self.user_activities_count[ins_id]['game_count']
        output['Clip'] = self.user_activities_count[ins_id]['Clip']
        output['Activity'] = self.user_activities_count[ins_id]['Activity']
        output['Assessment'] = self.user_activities_count[ins_id]['Assessment']
        output['Game'] = self.user_activities_count[ins_id]['Game']
        output['accumulated_actions'] = self.user_activities_count[ins_id]['all_actions_count']
        output['accumulated_correct_attempts'] \
            = self.user_activities_count[ins_id]['true_attempts_count']
        output['accumulated_uncorrect_attempts'] \
            = self.user_activities_count[ins_id]['false_attempts_count']
        evaluate_count = self.user_activities_count[ins_id]['evaluate_count']
        output['ave_accuracy'] = self.user_activities_count[ins_id]['accumulated_accracy'] / \
            evaluate_count if evaluate_count > 0 else 0
        output['ave_accuracy_group'] = self.user_activities_count[ins_id]['accumulated_accracy_group'] / \
            evaluate_count if evaluate_count > 0 else 0
        output['acc_group_0_count'] = self.user_activities_count[ins_id]['acc_group_0_count']
        output['acc_group_1_count'] = self.user_activities_count[ins_id]['acc_group_1_count']
        output['acc_group_2_count'] = self.user_activities_count[ins_id]['acc_group_2_count']
        output['acc_group_3_count'] = self.user_activities_count[ins_id]['acc_group_3_count']
        if len(self.durations[ins_id]) == 0:
            output['duration_mean'] = 0
        else:
            output['duration_mean'] = np.mean(self.durations[ins_id])
        return output

    def is_evaluate_timing(self, row, ass_stats):
        if not self.is_test:
            return (row['type'] == 'Assessment') and (row['end_of_game']) \
                and (ass_stats['step_count_in_game'] > 1)
        else:
            return (row['type'] == 'Assessment') and (row['end_of_game'])

    def is_labeled_timing(self, row):
        preview_stats = self.get_stats(row)
        preview_stats.update(self.assessment_acc.get_stats(row))
        preview_stats['step_count_in_game'] += 1
        session_type = row['type']
        session_titile = row['title']
        if session_type == 'Assessment' and row['event_code'] == self.win_code[session_titile]:
            if 'true' in row['event_data']:
                preview_stats['true_attempts_count'] += 1
                preview_stats['attempts_count'] += 1
            elif 'false' in row['event_data']:
                preview_stats['false_attempts_count'] += 1
                preview_stats['attempts_count'] += 1
        if self.is_test:
            return (session_type == 'Assessment') and (row['end_of_game'])
        else:
            return (session_type == 'Assessment') and (row['end_of_game']) \
                and (preview_stats['step_count_in_game'] > 1) and (preview_stats['attempts_count'] > 0)


class AssessmentAcc:
    def __init__(self, win_code):
        super().__init__()
        self.win_code = win_code
        self.acc = defaultdict(lambda: defaultdict(int))

    def update_acc(self, row):
        key = (row['installation_id'], row['game_session'])
        self.acc[key]['step_count_in_game'] += 1
        session_type = row['type']
        session_titile = row['title']
        if session_type == 'Assessment' and row['event_code'] == self.win_code[session_titile]:
            if 'true' in row['event_data']:
                self.acc[key]['true_attempts_count'] += 1
                self.acc[key]['attempts_count'] += 1
            elif 'false' in row['event_data']:
                self.acc[key]['false_attempts_count'] += 1
                self.acc[key]['attempts_count'] += 1

    def get_stats(self, row):
        key = (row['installation_id'], row['game_session'])
        output = {}
        output['accuracy_group'], output['accuracy'] = utils.get_accuracy_group(
            self.acc[key]['true_attempts_count'], self.acc[key]['false_attempts_count'])
        output['true_attempts_count'] = self.acc[key]['true_attempts_count']
        output['false_attempts_count'] = self.acc[key]['false_attempts_count']
        output['attempts_count'] = self.acc[key]['attempts_count']
        output['step_count_in_game'] = self.acc[key]['step_count_in_game']
        return output


class SessionAcc:
    def __init__(self):
        super().__init__()
        self.acc = defaultdict(lambda: defaultdict(list))

    def update_acc(self, row):
        key = (row['installation_id'], row['game_session'])
        timestamp = row['timestamp']
        self.acc[key]['time'].append(timestamp)

    def get_stats(self, row):
        key = (row['installation_id'], row['game_session'])
        output = {}
        time_list = self.acc[key]['time']
        output['session_time_length'] = (
            time_list[-1] - time_list[0]).seconds if len(time_list) > 0 else 0

        return output