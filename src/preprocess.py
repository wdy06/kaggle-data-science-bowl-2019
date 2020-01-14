import pandas as pd

import utils


def preprocess_dataset(dataset):
    # encode title
    activities_map = utils.load_json(utils.CONFIG_DIR / 'activities_map.json')
    world_map = utils.load_json(utils.CONFIG_DIR / 'world_map.json')
    dataset.main_df['title'] = dataset.main_df['title'].map(activities_map)
    dataset.main_df['world'] = dataset.main_df['world'].map(world_map)

    dataset.main_df['timestamp'] = pd.to_datetime(dataset.main_df['timestamp'])

    dataset.main_df.sort_values(['installation_id', 'timestamp'], inplace=True)

    dataset.main_df['end_of_game'] = dataset.main_df['game_session'].ne(
        dataset.main_df['game_session'].shift(-1).ffill())

    return dataset
