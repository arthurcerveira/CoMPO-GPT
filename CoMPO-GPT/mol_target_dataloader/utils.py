
def load_sets(set_path):
    file_paths = [set_path]
    if os.path.isdir(set_path):
        file_paths = sorted(glob.glob("{}/*.smi".format(set_path)))

    for path in it.cycle(file_paths):  # stores the path instead of the set
        return list(read_csv_file(path, num_fields=2))


def read_csv_file(file_path, ignore_invalid=True, num=-1, num_fields=0):

    with open_file(file_path, "rt") as csv_file:
        for i, row in enumerate(csv_file):
            if i == num:
                break
            fields = row.rstrip().split("\t")
            if fields:
                if num_fields > 0:
                    fields = fields[0:num_fields]
                yield fields
            elif not ignore_invalid:
                yield None


def open_file(path, mode="r", with_gzip=False):

    open_func = open
    if path.endswith(".gz") or with_gzip:
        open_func = gzip.open
    return open_func(path, mode)