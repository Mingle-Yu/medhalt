import pandas as pd
import json
import random
import os

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATASETS_FOLDER = os.path.join(CURRENT_FOLDER,"../datasets")

def read_json_(file):
    with open(file, 'r') as json_file:
        file_js = json.load(json_file)
    
    # -----调试代码-----
    print("==========prompts.utils: read_json starts==========")
    print(f"{file}: ", file_js)
    print("==========prompts.utils: read_json ends==========")
    # ------------------

    return file_js


prompt_dict = {
                'abs2pub' : './IR_abstract2pubmedlink/',
                'pmid2title' : './IR_pmid2title/',
                'url2title' : './IR_pubmedlink2title/',
                'title2pub' : './IR_title2pubmedlink/',
                'fake'      : './reasoning_Fake/',
                'FCT'       : './reasoning_FCT/',
                'Nota'      : './Reasoning_Nota/'}


data_dict = {
                'abs2pub' : 'IR_abstract2pubmedlink.csv',
                'pmid2title' : 'IR_pmid2title.csv',
                'url2title' : 'IR_pubmedlink2title.csv',
                'title2pub' : 'IR_title2pubmedlink.csv',
                'fake'      : 'reasoning_fake.csv',
                'FCT'       : 'reasoning_FCT.csv',
                'Nota'      : 'reasoning_nota.csv'
            }



def Nota_format(data_):
    # -----调试代码-----
    print("==========prompts.utils: Nota_format starts==========")
    print("data_: ", data_)
    # ------------------

    data_['prompt'] = data_.apply(lambda row: "Input: " + str({"Question": row['question'], "Options": eval(row['options'])}) + "\nOutput: ", axis=1)
    # -----调试代码-----
    print("data_: ", data_)
    print("==========prompts.utils: Nota_format ends==========")
    # ------------------

    return data_

def pmid2title_format(data_):
     # -----调试代码-----
    print("==========prompts.utils: pmid2title_format starts==========")
    print("data_: ", data_)
    # ------------------
    data_['prompt'] = data_.apply(lambda row: "Input: " + str({"Pmid": str(int(row["PMID"]))}), axis=1)
    # -----调试代码-----
    print("data_: ", data_)
    # ------------------
    data_['prompt'] = data_['prompt'].apply(lambda x: str(x) + "\n" + "Output: ")
    # -----调试代码-----
    print("data_: ", data_)
    print("==========prompts.utils: pmid2title_format ends==========")
    # ------------------
    return data_


def abs2pub_format(data_):
    # -----调试代码-----
    print("==========prompts.utils: abs2pub_format starts==========")
    print("data_: ", data_)
    # ------------------
    data_['prompt'] = data_.apply(lambda row: "Input: " + str({"paper_abstract": str(row["Abstract"])}), axis=1)
    # -----调试代码-----
    print("data_: ", data_)
    # ------------------
    data_['prompt'] = data_['prompt'].apply(lambda x: str(x) + "\n" + "Output: ")
    # -----调试代码-----
    print("data_: ", data_)
    print("==========prompts.utils: abs2pub_format ends==========")
    # ------------------
    return data_


def url2title_format(data_):
    # -----调试代码-----
    print("==========prompts.utils: url2title_format starts==========")
    print("data_: ", data_)
    # ------------------
    data_['prompt'] = data_.apply(lambda row: "Input: " + str({"url": str(row["url"])}), axis=1)
    # -----调试代码-----
    print("data_: ", data_)
    # ------------------
    data_['prompt'] = data_['prompt'].apply(lambda x: str(x) + "\n" + "Output: ")
    # -----调试代码-----
    print("data_: ", data_)
    print("==========prompts.utils: url2title_format ends==========")
    # ------------------
    return data_

def title2pub_format(data_):
    # -----调试代码-----
    print("==========prompts.utils: title2pub_format starts==========")
    print("data_: ", data_)
    # ------------------
    data_['prompt'] = data_.apply(lambda row: "Input: " + str({"paper_title": str(row["Title"])}), axis=1)
    # -----调试代码-----
    print("data_: ", data_)
    # ------------------
    data_['prompt'] = data_['prompt'].apply(lambda x: str(x) + "\n" + "Output: ")
    # -----调试代码-----
    print("data_: ", data_)
    print("==========prompts.utils: title2pub_format ends==========")
    # ------------------
    return data_


def get_samples(dataset_name, shots, prompt_version):
    # -----调试代码-----
    print("==========prompts.utils: get_samples starts==========")
    print("dataset_name: ", dataset_name, " shots: ", shots, " prompt_version: ", prompt_version)
    # ------------------

    prompt  = get_full_prompt(dataset_name, shots, prompt_version)

    # -----调试代码-----
    print("prompt after get_full_prompt: ", prompt)
    # ------------------

    dataset = load_dataset(dataset_name)

    # -----调试代码-----
    print("dataset after load_dataset: ", dataset)
    # ------------------

    dataset['prompt'] = dataset['prompt'].apply(lambda x: prompt + str(x))

    # -----调试代码-----
    print("dataset after adding prompt: ", dataset)
    # ------------------

    dataset = dataset.to_dict('records')

    # -----调试代码-----
    print("dataset after to_dict: ", dataset)
    print("==========prompts.utils: get_samples ends==========")
    # ------------------
    
    return dataset


def load_dataset(dataset_name):
    
    df = pd.read_csv(os.path.join(DATASETS_FOLDER,data_dict[dataset_name]))
    # -----调试代码-----
    print("==========prompts.utils: load_dataset starts==========")
    print("df: ", df)
    # ------------------
    if dataset_name == 'Nota' or dataset_name == 'FCT' or dataset_name == 'fake':
        df = Nota_format(df)
    
    elif dataset_name == 'pmid2title':
        df = pmid2title_format(df)
    
    elif dataset_name == 'abs2pub':
        df = abs2pub_format(df)
        
    elif dataset_name == 'url2title':
        df = url2title_format(df)
        
    elif dataset_name == 'title2pub':
        df = title2pub_format(df)

    # -----调试代码-----
    print("==========prompts.utils: load_dataset ends==========")
    # ------------------
    return df


def prompt_data(prompt_name, version, n_shots):
    # -----调试代码-----
    print("==========prompts.utils: prompt_data starts==========")
    # ------------------

    prompt    = read_json_(f"{os.path.join(CURRENT_FOLDER,prompt_dict[prompt_name],'prompts.json')}")['prompts']

    # -----调试代码-----
    print("prompt after read_json_: ", prompt)
    # ------------------

    prompt    = [prompt_ for prompt_ in prompt if prompt_['id'] == version][0]

    # -----调试代码-----
    print("prompt: ", prompt)
    # ------------------
    
    shots     = read_json_(f"{os.path.join(CURRENT_FOLDER,prompt_dict[prompt_name],'shots.json')}")['shots'][0]

    # -----调试代码-----
    print("shots after read_json_: ", shots)
    # ------------------

    default_p = [shot_ for shot_ in shots if shot_['prompt_type'] == 'default']

    # -----调试代码-----
    print("default shots: ", default_p)
    # ------------------

    task_p    = [shot_ for shot_ in shots if shot_['prompt_type'] != 'default']
    
    # -----调试代码-----
    print("task shots: ", task_p)
    print("==========prompts.utils: prompt_data ends==========")
    # ------------------
    
    if n_shots == 0:
        return {'prompt' : prompt, 'shots' : None}
    
    if n_shots == 1:
        return {'prompt' : prompt, 'shots' : random.sample(default_p, 1)}
    
    elif n_shots == 2:
        
        se_shots = random.sample(default_p, 1)
        other_p  = random.sample(task_p, 1)
        se_shots.extend(other_p)
        return {'prompt' : prompt, 'shots' : se_shots}
    
    elif n_shots == 3:
        
        se_shots = random.sample(default_p, 2)
        other_p  = random.sample(task_p, 1)
        se_shots.extend(other_p)
        return {'prompt' : prompt, 'shots' : se_shots}
    
    elif n_shots == 4:
        
        se_shots = random.sample(default_p, 2)
        other_p  = random.sample(task_p, 2)
        se_shots.extend(other_p)
        return {'prompt' : prompt, 'shots' : se_shots}
    
    elif n_shots == 5:
        
        se_shots = random.sample(default_p, 3)
        other_p  = random.sample(task_p, 2)
        se_shots.extend(other_p)
        return {'prompt' : prompt, 'shots' : se_shots}

    
def get_full_prompt(prompt_name, n_shots = 2, version = 'v0'):
    # -----调试代码-----
    print("==========prompts.utils: get_full_prompt starts==========")
    # ------------------

    prompt = prompt_data(prompt_name, version, n_shots)

    # -----调试代码-----
    print("prompt after prompt_data: ", prompt)
    print("==========prompts.utils: get_full_prompt ends==========")
    # ------------------

    if prompt['shots'] == None:
        return prompt['prompt']['prompt'] + '\n' + prompt['prompt']['output_format'] + '\n'
    else:
        all_examples = "Examples: \n"
        for sample in prompt['shots']:
            shot = f"Input : {sample['input']}\nOutput: {sample['Output']}Stop Here\n\n"
            all_examples+=shot
        final = prompt['prompt']['prompt'] + '\n' + prompt['prompt']['output_format'] + '\n' + all_examples
        return final

def get_sample_Dataset(n_shots, version):
    """
    [Deprecated] 该方法已弃用
    """
    df = pd.read_csv(os.path.join(DATASETS_FOLDER,'data_sample.csv'))
    df_d = df.to_dict('records')
    print("before_shape", len(df_d))
    
    total_changed = 0
    
    for index_, sample in enumerate(df_d):
        if sample['dataset_name'] == 'reasoning_fake':
            prompt_ = get_full_prompt('fake', n_shots, version)
            data_   = f"Input : {sample['qo']}\nOutput: " 
            full_input = prompt_ + data_
            df_d[index_]['prompt'] = full_input
            total_changed+=1

        elif sample['dataset_name'] == 'reasoning_nota':
            prompt_ = get_full_prompt('Nota', n_shots, version)
            data_   = f"Input : {sample['qo']}" 
            full_input = prompt_ + data_
            df_d[index_]['prompt'] = full_input
            total_changed+=1

        elif sample['dataset_name'] == 'reasoning_FCT':

            prompt_ = get_full_prompt('FCT', n_shots, version)
            data_   = f"Input : {sample['qo']}" 
            full_input = prompt_ + data_
            df_d[index_]['prompt'] = full_input
            total_changed+=1


        elif sample['dataset_name'] == 'IR_pubmedlink2title':

            prompt_ = get_full_prompt('url2title', n_shots, version)
            samplen = str({'url': sample['url']})
            data_   = f"Input: {samplen}\nOutput: "
            full_input = prompt_ + data_
            df_d[index_]['prompt'] = full_input
            total_changed+=1


        elif sample['dataset_name'] == 'IR_title2pubmedlink':

            prompt_ = get_full_prompt('title2pub', n_shots, version)
            samplen = str({'paper_title': sample['Title']})
            data_   = f"Input: {samplen}\nOutput: "
            full_input = prompt_ + data_
            df_d[index_]['prompt'] = full_input
            total_changed+=1

        elif sample['dataset_name'] == 'IR_pmid2title':

            prompt_ = get_full_prompt('pmid2title', n_shots, version)
            samplen = str({'Pmid': str(int(sample["PMID"]))})
            data_   = f"Input: {samplen}\nOutput: "
            full_input = prompt_ + data_
            df_d[index_]['prompt'] = full_input
            total_changed+=1

        elif sample['dataset_name'] == 'IR_abstract2pubmedlink':

            prompt_ = get_full_prompt('abs2pub', n_shots, version)
            samplen = str({"paper_abstract": sample["Abstract"]})
            data_   = f"Input: {samplen}\nOutput: "
            full_input = prompt_ + data_
            df_d[index_]['prompt'] = full_input
            total_changed+=1
    
    print("total_changed", total_changed)
    return pd.DataFrame(df_d)
