from collections import defaultdict

import numpy as np
from collections import Counter

import utils


class UserStatsAcc:
    def __init__(self, win_code, event_code_list, event_id_list, is_test):
        super().__init__()
        self.win_code = win_code
        self.event_code_list = event_code_list
        self.event_id_list = event_id_list
        self.ass_title_list = [12, 21,  2, 18, 30]
        self.event_feat_name = []
        self.title_feat_name = []
        # make feature name list
        for title in self.win_code.keys():
            self.title_feat_name.append(f'title{title}_count')
            for event_code in self.event_code_list:
                self.event_feat_name.append(
                    f'event_code{event_code}_count')
                self.event_feat_name.append(
                    f'title{title}_event{event_code}_count')
        self.event_feat_name += [
            f'id_{event_id}' for event_id in self.event_id_list]
        self.event_code_counter = defaultdict(lambda: Counter(
            {feat_name: 0 for feat_name in self.event_feat_name}))
        self.title_couter = defaultdict(lambda: Counter(
            {feat_name: 0 for feat_name in self.title_feat_name}))
        self.assessment_acc = AssessmentAcc(win_code)
        self.session_acc = SessionAcc(self.event_feat_name)
        self.user_activities_count = defaultdict(lambda: defaultdict(int))
        self.last_session_id = defaultdict(lambda: None)
        self.last_activity = defaultdict(int)
        self.last_title = defaultdict(lambda: None)
        self.last_title_acc = defaultdict(lambda: defaultdict(lambda: -1))
        self.last_world = defaultdict(lambda: None)
        self.last_time = {}
        self.durations = defaultdict(list)
        self.is_test = is_test

    def update_acc(self, row):
        ins_id = row['installation_id']
        session_id = row['game_session']
        session_type = row['type']
        session_title = row['title']
        if session_id != self.last_session_id.get(ins_id):
            self.user_activities_count[ins_id][session_type] += 1
        self.user_activities_count[ins_id]['all_actions_count'] += 1
        self.last_session_id[ins_id] = session_id
        self.last_activity[ins_id] = session_type
        # self.last_title[ins_id] = session_title
        # self.last_world[ins_id] = row['world']
        self.last_time[ins_id] = row['timestamp']
        # update chile accumulators
        self.assessment_acc.update_acc(row)
        self.session_acc.update_acc(row)
        ass_stats = self.assessment_acc.get_stats(row)
        session_stats = self.session_acc.get_stats(row)
        if row['end_of_game']:
            self.last_title[ins_id] = session_title
            self.last_world[ins_id] = row['world']
            self.user_activities_count[ins_id]['game_count'] += 1
            self.title_couter[ins_id][f'title{session_title}_count'] += 1
            self.event_code_counter[ins_id].update(
                self.session_acc.get_event_count(row))

            # delete data for memory usage
            self.session_acc.delete_data(row)

        if self.is_evaluate_timing(row, ass_stats):
            self.user_activities_count[ins_id][f'title{session_title}_evaluate_count'] += 1
            self.user_activities_count[ins_id][
                f'title{session_title}_true_attempts_count'] += ass_stats['true_attempts_count']
            self.user_activities_count[ins_id][
                f'title{session_title}_false_attempts_count'] += ass_stats['false_attempts_count']
            self.user_activities_count[ins_id][f'title{session_title}_accumulated_accracy'] += ass_stats['accuracy']
            self.user_activities_count[ins_id][
                f'title{session_title}_accumulated_accracy_group'] += ass_stats['accuracy_group']
            self.user_activities_count[ins_id]['evaluate_count'] += 1
            self.user_activities_count[ins_id]['true_attempts_count'] += ass_stats['true_attempts_count']
            self.user_activities_count[ins_id]['false_attempts_count'] += ass_stats['false_attempts_count']
            self.user_activities_count[ins_id]['accumulated_accracy'] += ass_stats['accuracy']
            self.user_activities_count[ins_id]['accumulated_accracy_group'] += ass_stats['accuracy_group']
            accuracy_group = ass_stats['accuracy_group']
            self.last_title_acc[ins_id][f'title{session_title}_last_acc_group'] = accuracy_group
            self.user_activities_count[ins_id][f'acc_group_{accuracy_group}_count'] += 1
            self.user_activities_count[ins_id][f'title{session_title}_acc_group_{accuracy_group}_count'] += 1
            self.durations[ins_id].append(session_stats['session_time_length'])

    def get_stats(self, row):
        ins_id = row['installation_id']
        # output = {}
        # output = {feat_name: self.user_activities_count[ins_id][feat_name] for feat_name in self.title_event_feat_name}
        output = dict(self.event_code_counter[ins_id])
        output.update(self.title_couter[ins_id])
        output['session_title'] = row['title']
        output['last_title'] = self.last_title[ins_id]
        output['last_world'] = self.last_world[ins_id]
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
            evaluate_count if evaluate_count > 0 else -1
        output['ave_accuracy_group'] = self.user_activities_count[ins_id]['accumulated_accracy_group'] / \
            evaluate_count if evaluate_count > 0 else -1
        output['acc_group_0_count'] = self.user_activities_count[ins_id]['acc_group_0_count']
        output['acc_group_1_count'] = self.user_activities_count[ins_id]['acc_group_1_count']
        output['acc_group_2_count'] = self.user_activities_count[ins_id]['acc_group_2_count']
        output['acc_group_3_count'] = self.user_activities_count[ins_id]['acc_group_3_count']
        for ass_title in self.ass_title_list:
            title_eval_count = self.user_activities_count[
                ins_id][f'title{ass_title}_evaluate_count']
            output[f'title{ass_title}_true_attempts_count'] = self.user_activities_count[ins_id][f'title{ass_title}_true_attempts_count']
            output[f'title{ass_title}_false_attempts_count'] = self.user_activities_count[ins_id][f'title{ass_title}_false_attempts_count']
            output[f'title{ass_title}_ave_accracy'] = self.user_activities_count[ins_id][f'title{ass_title}_accumulated_accracy'] / \
                title_eval_count if title_eval_count > 0 else -1
            output[f'title{ass_title}_ave_accracy_group'] = self.user_activities_count[ins_id][
                f'title{ass_title}_accumulated_accracy_group'] / title_eval_count if title_eval_count > 0 else -1
            output[f'title{ass_title}_acc_group_0_count'] = self.user_activities_count[ins_id][f'title{ass_title}_acc_group_0_count']
            output[f'title{ass_title}_acc_group_1_count'] = self.user_activities_count[ins_id][f'title{ass_title}_acc_group_1_count']
            output[f'title{ass_title}_acc_group_2_count'] = self.user_activities_count[ins_id][f'title{ass_title}_acc_group_2_count']
            output[f'title{ass_title}_acc_group_3_count'] = self.user_activities_count[ins_id][f'title{ass_title}_acc_group_3_count']
            output[f'title{ass_title}_last_acc_group'] = self.last_title_acc[ins_id][f'title{ass_title}_last_acc_group']
        if len(self.durations[ins_id]) == 0:
            output['duration_mean'] = 0
            output['duration_std'] = 0
        else:
            output['duration_mean'] = np.mean(self.durations[ins_id])
            output['duration_std'] = np.std(self.durations[ins_id])

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
    def __init__(self, event_feat_name_list):
        super().__init__()
        self.acc = defaultdict(lambda: defaultdict(list))
        self.event_feat_name_list = event_feat_name_list
        # self.event_counter = defaultdict(lambda: defaultdict(int))
        self.event_counter = defaultdict(lambda: Counter(
            {feat_name: 0 for feat_name in self.event_feat_name_list}))

    def update_acc(self, row):
        key = (row['installation_id'], row['game_session'])
        timestamp = row['timestamp']
        self.acc[key]['time'].append(timestamp)
        self.event_counter[key][f'event_code{row["event_code"]}_count'] += 1
        self.event_counter[key][f'title{row["title"]}_event{row["event_code"]}_count'] += 1
        self.event_counter[key][f'id_{row["event_id"]}'] += 1

    def get_stats(self, row):
        key = (row['installation_id'], row['game_session'])
        output = {}
        time_list = self.acc[key]['time']
        output['session_time_length'] = (
            time_list[-1] - time_list[0]).seconds if len(time_list) > 0 else 0

        return output

    def get_event_count(self, row):
        key = (row['installation_id'], row['game_session'])
        output = dict(self.event_counter[key])

        return output

    def delete_data(self, row):
        key = (row['installation_id'], row['game_session'])
        del self.event_counter[key]


class AssTitleAcc:
    def __init__(self, win_code):
        super().__init__()
        self.win_code = win_code
        self.ass_title_list = [12, 21, 2, 18, 30]
        self.acc = {}
        for ass_title in self.ass_title_list:
            self.acc[f'all_title{ass_title}_accuracy_list'] = []
            self.acc[f'all_title{ass_title}_accuracy_group_list'] = []
            self.acc[f'all_title{ass_title}_attempts_count_list'] = []
            self.acc[f'all_title{ass_title}_duration_list'] = []
        self.acc['tmp_true_attempts_count'] = 0
        self.acc['tmp_false_attempts_count'] = 0
        self.acc['tmp_attempts_count'] = 0

    def update_acc(self, row):
        session_type = row['type']
        session_title = row['title']
        if session_type == 'Assessment' and row['event_code'] == self.win_code[session_title]:
            # if session_title == 30:
            #     print(session_type, session_title)
            #     print(self.acc)
            if 'true' in row['event_data']:
                self.acc['tmp_true_attempts_count'] += 1
                self.acc['tmp_attempts_count'] += 1
            elif 'false' in row['event_data']:
                self.acc['tmp_false_attempts_count'] += 1
                self.acc['tmp_attempts_count'] += 1
        # if row['end_of_game'] and row['type'] == 'Assessment':
            if row['end_of_game']:
                true_attempts = self.acc['tmp_true_attempts_count']
                false_attempts = self.acc['tmp_false_attempts_count']
                attempts_count = self.acc['tmp_attempts_count']
                accuracy_group, accuracy = utils.get_accuracy_group(
                    true_attempts, false_attempts)
                self.acc[f'all_title{session_title}_accuracy_list'].append(
                    accuracy)
                self.acc[f'all_title{session_title}_accuracy_group_list'].append(
                    accuracy_group)
                self.acc[f'all_title{session_title}_attempts_count_list'].append(
                    attempts_count)
                self.acc[f'all_title{session_title}_duration_list'].append(
                    row['game_time'] / 1000)

                # reset some counter
                # print('----------------------------')
                self.acc['tmp_true_attempts_count'] = 0
                self.acc['tmp_false_attempts_count'] = 0
                self.acc['tmp_attempts_count'] = 0

    def get_stats(self):
        output = {}
        for ass_title in self.ass_title_list:
            output[f'all_title{ass_title}_accuracy_mean'] = np.mean(
                self.acc[f'all_title{ass_title}_accuracy_list'])
            output[f'all_title{ass_title}_accuracy_std'] = np.std(
                self.acc[f'all_title{ass_title}_accuracy_list'])
            output[f'all_title{ass_title}_accuracy_group_mean'] = np.mean(
                self.acc[f'all_title{ass_title}_accuracy_group_list'])
            output[f'all_title{ass_title}_accuracy_group_std'] = np.std(
                self.acc[f'all_title{ass_title}_accuracy_group_list'])
            output[f'all_title{ass_title}_attempts_count_mean'] = np.mean(
                self.acc[f'all_title{ass_title}_attempts_count_list'])
            output[f'all_title{ass_title}_attempts_count_std'] = np.std(
                self.acc[f'all_title{ass_title}_attempts_count_list'])
            output[f'all_title{ass_title}_duration_mean'] = np.mean(
                self.acc[f'all_title{ass_title}_duration_list'])
            output[f'all_title{ass_title}_duration_std'] = np.std(
                self.acc[f'all_title{ass_title}_duration_list'])

        return output

    def get_mapper(self):
        # title_stats = self.get_stats()
        map_dict = {}
        map_dict['user_accuracy_mean'] = {}
        map_dict['user_accuracy_std'] = {}
        map_dict['user_accuracy_group_mean'] = {}
        map_dict['user_accuracy_group_std'] = {}
        map_dict['user_attempts_count_mean'] = {}
        map_dict['user_attempts_count_std'] = {}
        map_dict['user_duration_mean'] = {}
        map_dict['user_duration_std'] = {}
        for title in self.ass_title_list:
            map_dict['user_accuracy_mean'][title] = np.mean(
                self.acc[f'all_title{title}_accuracy_list'])
            map_dict['user_accuracy_std'][title] = np.std(
                self.acc[f'all_title{title}_accuracy_list'])
            map_dict['user_accuracy_group_mean'][title] = np.mean(
                self.acc[f'all_title{title}_accuracy_group_list'])
            map_dict['user_accuracy_group_std'][title] = np.std(
                self.acc[f'all_title{title}_accuracy_group_list'])
            map_dict['user_attempts_count_mean'][title] = np.mean(
                self.acc[f'all_title{title}_attempts_count_list'])
            map_dict['user_attempts_count_std'][title] = np.std(
                self.acc[f'all_title{title}_attempts_count_list'])
            map_dict['user_duration_mean'][title] = np.mean(
                self.acc[f'all_title{title}_duration_list'])
            map_dict['user_duration_std'][title] = np.std(
                self.acc[f'all_title{title}_duration_list'])

        return map_dict
