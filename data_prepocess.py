import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
    df = df[['altitude', 'indicated_airspeed', 'roll', 'pitch']]
    df = df.astype(numpy.float64)

    df['pitch(t - 1)'] = numpy.nan

    for i in range(1, df.shape[0]):
        df.loc[i, 'pitch(t - 1)'] = df.loc[i - 1, 'pitch']

    fp.close()

    return df.drop(0, axis=0)


def normalization(df):
    scaler = MinMaxScaler()

    scaler.fit(df[['altitude', 'indicated_airspeed', 'roll', 'pitch(t - 1)']])
    df[['altitude', 'indicated_airspeed', 'roll', 'pitch(t - 1)']] = scaler.transform(
        df[['altitude', 'indicated_airspeed', 'roll', 'pitch(t - 1)']])

    # df.drop('pitch', axis=1).to_csv('./predict.csv', index=False)
    inputs = df.drop('pitch', axis=1)
    targets = df['pitch']
    return inputs, targets, scaler

def split_data(inputs, targets, train_size, random_state):
    return train_test_split(inputs, targets,
                            train_size=train_size, random_state=random_state)



if __name__ == '__main__':
    fp = open('./input_fl_12477', mode='r', encoding='UTF-8')
    columns = []
    data = []
    for index, row in enumerate(fp.readlines()):

        if index == 0:
            columns.extend(row.split(' '))
        else:
            data.append(row.strip().split(' '))

    df = pandas.DataFrame(data=data, columns=columns)
    df = df[['altitude', 'indicated_airspeed', 'roll', 'pitch']]
    df = df.astype(numpy.float64)

    df['pitch(t - 1)'] = numpy.nan

    for i in range(1, df.shape[0]):
        df.loc[i, 'pitch(t - 1)'] = df.loc[i - 1, 'pitch']

    df.drop(0, axis=0, inplace=True)
    df.to_csv('./input_fl_12477.csv', index=False)

    fp.close()
