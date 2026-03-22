from itertools import islice

def create_short_file(r_file_path :str, limit :int, w_file_path:str):
    with open(r_file_path, 'r') as file:
        lines=[line for line in islice(file,limit)]
        content=''.join(line for line in lines)
    file.close()

    with open(w_file_path, 'w') as f:
        f.write(content)
    f.close