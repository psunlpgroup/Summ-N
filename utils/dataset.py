import os.path


def write_list_asline(path, data):
    with open(path,'w',encoding='utf-8') as file:
        for sample in data:
            file.write(sample.strip() + '\n')

def read_list_asline(path):
    data = []
    with open(path,'r',encoding='utf-8')  as file:
        for line in file:
            data.append(line.strip())
    return data


def assertion_statis(source_data, target_data, prompt):
    assert len(source_data['train']) == len(target_data['train'])
    assert len(source_data['val']) == len(target_data['val'])
    assert len(source_data['test']) == len(target_data['test'])

    print(prompt)
    print("Train size:", len(source_data['train']))
    print("Val size:", len(source_data['val']))
    print("Test size:", len(source_data['test']))


def write_finegrained_dataset(source, target, stage_folder):
    if os.path.exists(stage_folder) is False:
        os.makedirs(stage_folder)
    for data_type in ['train','val','test']:
        source_path = os.path.join(stage_folder, f"{data_type}.source")
        write_list_asline(source_path, source[data_type])

        target_path = os.path.join(stage_folder, f"{data_type}.target")
        write_list_asline(target_path, target[data_type])

# Load three splits in the data folder with certain suffix
def load_split_aslist(folder_path, suffix='source'):
    data = {}
    for data_type in ['train', 'test', 'val']:
        split_path = os.path.join(folder_path, f"{data_type}.{suffix}")
        if os.path.exists(split_path):
            data[data_type] = read_list_asline(split_path)
        else:
            data[data_type] = []
    return data