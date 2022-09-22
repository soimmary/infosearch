import os


def get_directories(main_dir: str) -> list:
    all_dirs = []
    seasons = os.listdir(main_dir)
    for season in seasons:
        dirs = os.listdir(f'{main_dir}/{season}')
        for dir in dirs:
            all_dirs.append(f'{main_dir}/{season}/{dir}')
    return all_dirs
