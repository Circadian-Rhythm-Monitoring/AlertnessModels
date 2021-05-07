import pandas as pd
import os


def split_by_day(file_path, ignore_days=2):
    df = pd.read_csv(file_path)
    print(df.tail())

    ignore_rows = ignore_days*24
    start_flag = False

    ret = dict()
    day = 0
    day_complete = True
    one_flag = False
    day_df = None
    all_beta = []
    for idx, row in df.iterrows():
        if idx < ignore_rows:
            continue
        beta = row['beta']
        all_beta.append(beta)
        if beta == 1:
            one_flag = True
            one_flag = True
        if not start_flag and not one_flag and beta == 0:
            continue
        if not start_flag and beta == 1:
            continue
        if not start_flag and beta == 0:
            start_flag = True
        if not start_flag:
            continue

        if all_beta[-2] == 1 and all_beta[-1] == 0:
            day_complete = True
        if day_complete:
            if day_df is not None:
                day += 1
            day_complete = False
            if day_df is not None:
                n_ts = day_df.shape[0]
                col_day = [day] * n_ts
                day_df['day'] = col_day
                ret[day] = day_df
            day_df = pd.DataFrame()
        day_df = day_df.append(row, ignore_index=True)

    return ret


def insert_row(df, row_number, row_value):
    start_upper = 0
    end_upper = row_number
    start_lower = row_number
    end_lower = df.shape[0]

    upper_half = [*range(start_upper, end_upper)]
    lower_half = [*range(start_lower, end_lower)]
    lower_half = [x+1 for x in lower_half]

    new_index = upper_half + lower_half
    df.index = new_index
    df.loc[row_number] = row_value
    df = df.sort_index()

    return df


def scale_data(data, scale=24):
    var_names = ['x', 'beta', 'H', 'B', 'W', 'sleepness', 'subjective_alertness', 'dH', 'dW', 'ts', 'xc']
    ret = dict()
    for day in data.keys():
        day_df = data[day]
        df_len = day_df.shape[0]

        # if df_len < 15:
        #     continue

        if df_len > scale:
            diff = df_len - scale
            drop_step_length = int(df_len / (diff + 1))
            drop_idx = [drop_step_length*(i+1) for i in range(diff)]
            day_df.drop(drop_idx, inplace=True)
            day_df.reset_index(drop=True, inplace=True)
            ret[day] = day_df
        elif df_len == scale:
            ret[day] = day_df
        else:
            diff = scale - df_len
            add_step_length = int(df_len / (diff + 1))
            add_idx = [add_step_length*(i+1) for i in range(diff)]
            add_idx.sort(reverse=True)
            for idx in add_idx:
                row1 = day_df.iloc[idx]
                # row2 = day_df.iloc[idx+1]
                # new_row = dict()
                # for var in var_names:
                #     new_row[var] = (row1[var] + row2[var]) / 2
                new_row = row1
                day_df = insert_row(day_df, idx+1, new_row)
            ret[day] = day_df
    return ret


if __name__ == '__main__':
    data_dict = split_by_day('data/data.csv')
    # data = scale_data(data_dict)

    all_df = pd.DataFrame()
    for day in data_dict.keys():
        if not os.path.isdir('data/scaled/'):
            os.makedirs('data/scaled/')
        # data_dict[day].to_csv('data/scaled/{}.csv'.format(day), index=False)
        all_df = pd.concat([all_df, data_dict[day]], ignore_index=False)
    all_df.to_csv('data/train.csv', index=True, index_label='day_ts')
