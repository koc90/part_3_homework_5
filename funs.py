import pandas as pd
from matplotlib import pyplot as plt
import os


ACTIVITIES = ["idle", "stairs", "walking", "running"]


def read_data_from_folder(folder_name):
    dirname = os.path.join("data", folder_name)
    i = 0
    for f in os.listdir(dirname):
        path = os.path.join("data", folder_name, f)
        if i == 0:
            df = pd.read_csv(path)
        else:
            new_df = pd.read_csv(path)
            df = pd.concat([df, new_df])
        i += 1
    return df


def create_list_of_files_in_folder(folder_name):
    dirname = os.path.join("data", folder_name)
    list_of_files = os.listdir(dirname)
    return list_of_files


def create_data_file_path(folder_name, file_name):
    path = os.path.join("data", folder_name, file_name)
    return path


def read_data_from_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def rms(df: pd.DataFrame) -> pd.DataFrame:
    l = df.shape[0]
    res = ((df.pow(2).sum()) / l).pow(0.5)
    return res


def test_rms():
    activity_num = 0
    activity_name = ACTIVITIES[activity_num]
    list_of_files = create_list_of_files_in_folder(activity_name)
    path = create_data_file_path(activity_name, list_of_files[0])
    df = read_data_from_file(path)
    rms(df)


def calculate_time_domain_features_for_one_sample(
    df: pd.DataFrame, cls: int
) -> pd.DataFrame:
    one_sample_features = dict()
    axes = ["X", "Y", "Z"]
    features = {
        "mean": df.mean(axis=0),
        "variance": df.var(axis=0),
        "standard_deviation": df.std(axis=0),
        "median": df.median(axis=0),
        "max": df.max(axis=0),
        "min": df.min(axis=0),
        "rms": rms(df),
    }
    for key, value in features.items():
        names = ["_".join([key, axis]) for axis in axes]

        for i in range(3):
            one_sample_features[names[i]] = [value.iloc[i]]
    one_sample_features["activity"] = cls
    new_df = pd.DataFrame(one_sample_features)
    return new_df


def calculate_time_domain_features_all_in_folder(folder_name, activity_num):
    files = create_list_of_files_in_folder(folder_name)
    file0 = files[0]
    path0 = create_data_file_path(folder_name, file0)
    df0 = read_data_from_file(path0)
    new_df = calculate_time_domain_features_for_one_sample(df0, activity_num)
    time_domain_features = new_df.drop(index=0)
    for file in files:
        path = create_data_file_path(folder_name, file)
        df = read_data_from_file(path)
        new_df = calculate_time_domain_features_for_one_sample(df, activity_num)
        time_domain_features = pd.concat(
            [time_domain_features, new_df], ignore_index=True
        )
    return time_domain_features


def plot_signal(df: pd.DataFrame, activity_name: str, sample_num: int):
    time = list(df.index)
    a_X = list(df[df.columns[0]])
    a_Y = list(df[df.columns[1]])
    a_Z = list(df[df.columns[2]])
    plt.plot(time, a_X, "b")
    plt.plot(time, a_Y, "r")
    plt.plot(time, a_Z, "g")
    plt.legend(["a_X", "a_Y", "a_Z"])
    plt.title(f"{activity_name} number {sample_num}")
    plt.xlabel("'time'")
    plt.grid(True)
    plt.show()


def show_few_pictures(activity_num, samples_num):
    for sample_num in samples_num:
        activity_name = ACTIVITIES[activity_num]
        test_path = create_data_file_path(
            activity_name, create_list_of_files_in_folder(activity_name)[sample_num]
        )
        test_df = read_data_from_file(test_path)
        plot_signal(test_df, activity_name, sample_num)


def main():
    pass


if __name__ == "__main__":
    main()
