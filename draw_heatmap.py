import pandas
import seaborn as sns
import matplotlib.pyplot as plt

def feature_selection(path):
    fp = open(path, mode='r', encoding='UTF-8')
    columns = []
    data = []
    for index, row in enumerate(fp.readlines()):

        if index == 0:
            columns.extend(row.split(' '))
        else:
            data.append(row.strip().split(' '))

    df = pandas.DataFrame(data=data, columns=columns)
    df = df.drop(['id', 'flight', 'phase', 'nav_1_freq', 'nav_2_freq', 'vertical_acceleration'], axis=1)

    return df.drop(0, axis=0)

if __name__ == '__main__':
    df = feature_selection('data/input_fl_12477')

    corr = df.corr()

    plt.figure(figsize=(25, 16))
    sns.heatmap(corr)
    plt.savefig('./ScreenShoot/corr')