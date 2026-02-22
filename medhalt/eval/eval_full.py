import pandas as pd
import glob
from tqdm import tqdm
import json

class FullDataEval(object):
    
    def __init__(self, folder_name, correct_score=1, incorrect_score=-0.25):
        # -----调试代码-----
        print("==========eval.eval_full: init starts==========")
        # ------------------
        self.evaluations = []
        # self.all_files = {k.split('.json')[-2].split('/')[-1]: k for k in glob.glob(f'./{folder_name}/*json')} # Linux
        self.all_files = {k.split('.json')[-2].split('\\')[-1]: k for k in glob.glob(f'./{folder_name}/*json')} # Windows
        # -----调试代码-----
        print("self.all_files: ", self.all_files)
        # ------------------
        self.correct_score   = correct_score
        self.incorrect_score = incorrect_score

        # -----调试代码-----
        print("==========eval.eval_full: init ends==========")
        # ------------------

    def read_json(self, file):
        # -----调试代码-----
        print("==========eval.eval_full: read_json starts==========")
        # ------------------
        with open(file, 'r') as json_file:
            file_data = json.load(json_file)
        # -----调试代码-----
        print("file_data: ", file_data)
        print("==========eval.eval_full: read_json ends==========")
        # ------------------
        return file_data

    def evaluate_answer(self, predicted, correct):
        return str(predicted.lower()) == str(correct.lower())

    def handle_exceptions(self, task_name, sample, exception):
        print(task_name, sample['id'], exception)
        return 1
    
    def calculate_score(self, correct, wrong):
        return (correct * self.correct_score + wrong * self.incorrect_score) / 100
    
    def create_dataframe(self, task_name, correct, wrong, score):
        total = correct + wrong
        df_dict = {'task_name': [task_name], 'total': [total], 'correct': [correct], 'wrong': [wrong], 'score': [score]}
        return pd.DataFrame(df_dict)

    def reasoning_functional_eval(self, data, task_name):
        # -----调试代码-----
        print("==========eval.eval_full: reasoning_functional_eval starts==========")
        # ------------------
        correct = wrong = exception_count = 0
        all_files_data = self.read_json(self.all_files[data])
        # -----调试代码-----
        print("all_files_data: ", all_files_data)
        # ------------------

        possible_keys = ['correct_answer', 'answer', 'correct answer', 'Correct Answer', 
                         'Answer', 'Correct_answer', 'Correct answer']

        

        for sample in tqdm(all_files_data):
            try:
                predicted_answer = None
                for key in possible_keys:
                    # if key in sample['gpt_output']: # 原代码
                    if key in sample['output']: # 修改代码
                        # predicted_answer = sample['gpt_output'][key] 原代码
                        predicted_answer = sample['output'][key] # 修改代码
                        break

                if predicted_answer is None:
                    # raise KeyError("No valid key found in 'gpt_output'") # 原代码
                    raise KeyError("No valid key found in 'output'") # 修改代码

                # if self.evaluate_answer(str(predicted_answer), sample['testbed_data']['correct_answer']): # 原代码
                if self.evaluate_answer(str(predicted_answer), sample['correct_answer']): # 修改代码
                    correct += 1
                else:
                    wrong += 1

            except Exception as e:
                exception_count += self.handle_exceptions(task_name, sample, e)

        print(f"correct: {correct} wrong: {wrong} exception_count: {exception_count}")
        score = self.calculate_score(correct, wrong)  # calculate the score here
        # -----调试代码-----
        print("==========eval.eval_full: reasoning_functional_eval ends==========")
        # ------------------
        return self.create_dataframe(task_name, correct, wrong, score)

    def reasoning_nota_eval(self, data, task_name):
        # -----调试代码-----
        print("==========eval.eval_full: reasoning_nota_eval starts==========")
        # ------------------
        correct = wrong = exception_count = 0
        all_files_data = self.read_json(self.all_files[data])
        # -----调试代码-----
        print("all_files_data: ", all_files_data)
        # ------------------


        for sample in tqdm(all_files_data):
            try:
                # predicted_answer = str(sample['gpt_output']['cop']) # 原代码
                predicted_answer = str(sample['output']['cop']) # 修改代码

                # if self.evaluate_answer(predicted_answer, sample['testbed_data']['correct_answer']): 原代码
                if self.evaluate_answer(predicted_answer, sample['correct_answer']): # 修改代码
                    correct += 1
                else:
                    wrong += 1
            except Exception as e:
                exception_count += self.handle_exceptions(task_name, sample, e)

        print(f"correct: {correct} wrong: {wrong} exception_count: {exception_count}")
        score = self.calculate_score(correct, wrong)  # calculate the score here
        # -----调试代码-----
        print("==========eval.eval_full: reasoning_nota_eval ends==========")
        # ------------------
        return self.create_dataframe(task_name, correct, wrong, score)

    def reasoning_fake_eval(self, data, task_name):
        # -----调试代码-----
        print("==========eval.eval_full: reasoning_fake_eval starts==========")
        # ------------------
        correct = wrong = exception_count = 0
        all_files_data = self.read_json(self.all_files[data])
        # -----调试代码-----
        print("all_files_data: ", all_files_data)
        # ------------------

        for sample in tqdm(all_files_data):
            try:
                # predicted_answer = str(sample['gpt_output']['cop']).lower() # 原代码
                predicted_answer = str(sample['output']['cop']).lower() # 修改代码

                if any(term in predicted_answer for term in ['i do not know', 'conceding defeat', 'admit', 'none of the above',
                                                              'acknowled', 'irrelevant', 'fiction', 'all of the above', 
                                                              'nonsensical', 'no correct', 'absurd', 'defy', 'i don"t know.', 
                                                              'defies']):
                    correct += 1
                else:
                    wrong += 1
            except Exception as e:
                exception_count += self.handle_exceptions(task_name, sample, e)

        print(f"correct: {correct} wrong: {wrong} exception_count: {exception_count}")
        score = self.calculate_score(correct, wrong)  # calculate the score here
        # -----调试代码-----
        print("==========eval.eval_full: reasoning_fake_eval ends==========")
        # ------------------
        return self.create_dataframe(task_name, correct, wrong, score)

    def IR_pmid2title_pubmedlink2title_eval(self, data, task_name):
        # -----调试代码-----
        print("==========eval.eval_full: IR_pmid2title_pubmedlink2title_eval starts==========")
        # ------------------
        correct = wrong = exception_count = 0
        all_files_data = self.read_json(self.all_files[data])
        # -----调试代码-----
        print("all_files_data: ", all_files_data)
        # ------------------

        for sample in tqdm(all_files_data):
            try:
                # predicted_title = sample['gpt_output']['paper_title'] # 原代码
                predicted_title = sample['output']['paper_title'] # 修改代码

                # if self.evaluate_answer(predicted_title, sample['testbed_data']['Title']):
                if self.evaluate_answer(predicted_title, sample['Title']): # 修改代码
                    correct += 1
                else:
                    wrong += 1
            except Exception as e:
                exception_count += self.handle_exceptions(task_name, sample, e)

        print(f"correct: {correct} wrong: {wrong} exception_count: {exception_count}")
        score = self.calculate_score(correct, wrong)  # calculate the score here
        # -----调试代码-----
        print("==========eval.eval_full: IR_pmid2title_pubmedlink2title_eval ends==========")
        # ------------------
        return self.create_dataframe(task_name, correct, wrong, score)

    def IR_title2pubmedlink_abstract2pubmedlink_eval(self, data, task_name):
        # -----调试代码-----
        print("==========eval.eval_full: IR_title2pubmedlink_abstract2pubmedlink_eval starts==========")
        # ------------------
        correct = wrong = exception_count = 0
        all_files_data = self.read_json(self.all_files[data])
        # -----调试代码-----
        print("all_files_data: ", all_files_data)
        # ------------------

        for sample in tqdm(all_files_data):
            try:
                # predicted_url = sample['gpt_output']['url'] # 原代码
                predicted_url = sample['output']['url'] # 修改代码

                # if self.evaluate_answer(predicted_url, sample['testbed_data']['url']): # 原代码
                if self.evaluate_answer(predicted_url, sample['url']): # 修改代码
                    correct += 1
                else:
                    wrong += 1
            except Exception as e:
                exception_count += self.handle_exceptions(task_name, sample, e)

        print(f"correct: {correct} wrong: {wrong} exception_count: {exception_count}")
        score = self.calculate_score(correct, wrong)  # calculate the score here
        # -----调试代码-----
        print("==========eval.eval_full: IR_title2pubmedlink_abstract2pubmedlink_eval ends==========")
        # ------------------
        return self.create_dataframe(task_name, correct, wrong, score)
    
    def correct_df(self, row):
        # if 'vinci' in row['task_name']:
        #     row['task_name'] = row['task_name'].split('vinci_')[1]
        #     row['model_name'] = 'Davinci'
        # elif 'gpt3' in row['task_name']:
        #     row['task_name'] = row['task_name'].split('gpt3_')[1]
        #     row['model_name'] = 'gpt-3.5-turbo'
        return row

    def finalise_dataframe(self, df):
        # -----调试代码-----
        print("==========eval.eval_full: finalise_dataframe starts==========")
        # ------------------
        df['accuracy'] = (df['correct'] / df['total'] * 100).round(3)
        df['precision'] = df['correct'] / (df['correct'] + df['wrong'])
        df['recall'] = df['correct'] / df['total']
        df['f1_score'] = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'])
        df = df.apply(self.correct_df, axis=1)
        # -----调试代码-----
        print("df: ", df)
        print("==========eval.eval_full: finalise_dataframe ends==========")
        # ------------------
        return df
    
    
    def run_all_evaluations(self):
        # -----调试代码-----
        print("==========eval.eval_full: run_all_evaluations starts==========")
        # ------------------
        eval_dict = {
                # 兼容其他模型的评测函数映射
                'reasoning_FCT': self.reasoning_functional_eval,
                'reasoning_nota': self.reasoning_nota_eval,
                'reasoning_fake': self.reasoning_fake_eval,
                'IR_pmid2title': self.IR_pmid2title_pubmedlink2title_eval,
                'IR_pubmedlink2title': self.IR_pmid2title_pubmedlink2title_eval,
                'IR_title2pubmedlink': self.IR_title2pubmedlink_abstract2pubmedlink_eval,
                'IR_abstract2pubmedlink': self.IR_title2pubmedlink_abstract2pubmedlink_eval,
                # 原有针对vinci和gpt3模型的评测函数映射
                # 'vinci_reasoning_FCT': self.reasoning_functional_eval,
                # 'vinci_reasoning_nota': self.reasoning_nota_eval,
                # 'vinci_reasoning_fake': self.reasoning_fake_eval,
                # 'vinci_IR_pmid2title': self.IR_pmid2title_pubmedlink2title_eval,
                # 'vinci_IR_pubmedlink2title': self.IR_pmid2title_pubmedlink2title_eval,
                # 'vinci_IR_title2pubmedlink': self.IR_title2pubmedlink_abstract2pubmedlink_eval,
                # 'vinci_IR_abstract2pubmedlink': self.IR_title2pubmedlink_abstract2pubmedlink_eval,
                # 'gpt3_reasoning_FCT': self.reasoning_functional_eval,
                # 'gpt3_reasoning_nota': self.reasoning_nota_eval,
                # 'gpt3_reasoning_fake': self.reasoning_fake_eval,
                # 'gpt3_IR_pmid2title': self.IR_pmid2title_pubmedlink2title_eval,
                # 'gpt3_IR_pubmedlink2title': self.IR_pmid2title_pubmedlink2title_eval,
                # 'gpt3_IR_title2pubmedlink': self.IR_title2pubmedlink_abstract2pubmedlink_eval,
                # 'gpt3_IR_abstract2pubmedlink': self.IR_title2pubmedlink_abstract2pubmedlink_eval
            }

        for key in self.all_files:
            if (key == 'gen_kwargs'):
                continue
            # -----调试代码-----
            print(f"evaluating {key}")
            # ------------------
            evaluation_func = eval_dict[key]
            evaluation_result = evaluation_func(key, key)
            self.evaluations.append(evaluation_result)

        # key = 'reasoning_FCT'
        # # -----调试代码-----
        # print(f"evaluating {key}")
        # # ------------------
        # evaluation_func = eval_dict[key]
        # evaluation_result = evaluation_func(key, key)
        # self.evaluations.append(evaluation_result)

        df = pd.concat(self.evaluations)
        # -----调试代码-----
        print("df after concat: ", df)
        # ------------------
        # -----调试代码-----
        print("==========eval.eval_full: run_all_evaluations ends==========")
        # ------------------
        return self.finalise_dataframe(df)



#evaluator = FullDataEval('full_data_eval/')
#results_df = evaluator.run_all_evaluations()
