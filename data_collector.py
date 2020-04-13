import consts
import pandas as pd


# collect_url_csv_data(consts.URLS.EUROPE_LOCATIONS, consts.PATHS.EUROPE_LOCATIONS)
# collect_url_csv_data(consts.URLS.EUROPE_COVID19_STATISTICS, consts.PATHS.EUROPE_COVID19_STATISTICS)
def collect_url_csv_data(url, path_to_save):
    import requests
    from os import path, remove

    file_to_save = requests.get(url)

    if path.exists(path_to_save):
        remove(path_to_save)

    open(path_to_save, 'wb').write(file_to_save.content)


def make_plot_country_comparison(col_name, countries):
    ax = plt.gca()
    for c in countries:
        c.plot(kind='line', x='date', y=col_name, ax=ax, label=c['location'].iloc[1])
    plt.savefig(consts.PATHS.PLOT_DIR + col_name + '_country_comparison.png')
    plt.clf()


def make_country_statistic_plot(col_name, c1):
    ax = plt.gca()
    c1.plot(kind='line', x='date', y=col_name[0], color='yellow', ax=ax)
    c1.plot(kind='line', x='date', y=col_name[1], color='brown', ax=ax)
    c1.plot(kind='line', x='date', y=col_name[2], color='orange', ax=ax)
    c1.plot(kind='line', x='date', y=col_name[3], color='black', ax=ax)
    plt.title(c1['location'].iloc[1])
    plt.savefig(consts.PATHS.PLOT_DIR + c1['location'].iloc[1] + '.png')
    plt.clf()


if __name__ == "__main__":

    df = pd.read_csv(consts.PATHS.EUROPE_LOCATIONS)
    germany_data = df[df['location'] == 'Germany']
    italy_data = df[df['location'] == 'Italy']
    china_data = df[df['location'] == 'China']

    import matplotlib.pyplot as plt
    import pandas as pd

    # gca stands for 'get current axis'

    columns = ['new_cases', 'new_deaths', 'total_cases', 'total_deaths']
    for col in columns:
        make_plot_country_comparison(col, [germany_data, italy_data, china_data])

    for c in [germany_data, italy_data, china_data]:
        make_country_statistic_plot(columns, c)
