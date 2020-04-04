def collect_url_csv_data(url, path_to_save):
    import requests
    from os import path, remove

    file_to_save = requests.get(url)

    if path.exists(path_to_save):
        remove(path_to_save)

    open(path_to_save, 'wb').write(file_to_save.content)


if __name__ == "__main__":
    import consts

    collect_url_csv_data(consts.URLS.EUROPE_LOCATIONS, consts.DATA_PATH.EUROPE_LOCATIONS)
    collect_url_csv_data(consts.URLS.EUROPE_COVID19_STATISTICS, consts.DATA_PATH.EUROPE_COVID19_STATISTICS)
