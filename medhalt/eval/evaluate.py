import pandas as pd
from medhalt.eval.eval_full import FullDataEval
import glob,os,json
import ast,re,numpy as np

pred_prefix_dict = {
    'abs2pub':  'IR_abstract2pubmedlink',
    'pmid2title' : 'IR_pmid2title',
    'url2title' : 'IR_pubmedlink2title',
    'title2pub' : 'IR_title2pubmedlink',
    'fake'      : 'reasoning_fake',
    'FCT'       : 'reasoning_FCT',
    'Nota'      : 'reasoning_nota'
}

ds_name_dict = {v:k for k,v in pred_prefix_dict.items()}

def escaped_(data: str):
    # -----调试代码-----
    print("==========eval.evaluate: escaped starts==========")
    print("data before escaping: ", data)
    # ------------------
    if "'" in data:
        escaped_str = re.sub(r"(?<=\w)(')(?=\w)", r"\"", data)
    else:
        escaped_str = re.sub(r'(?<=\w)(")(?=\w)', r"\'", data)
    
    # -----调试代码-----
    print("data before escaping: ", escaped_str)
    print("==========eval.evaluate: escaped ends==========")
    # ------------------

    return escaped_str

def parse_key_values(out_str):
    #regex = r"""['"]{key}['"]\s*:\s*['"]*(.*?)['"]*\s*[,}}]""".format(key=key)
    regex = r"""['"](.*?)['"]\s*:\s*['"]*(.*?)['"]*\s*[,}]"""
    regex = re.compile(regex)
    return regex.findall(out_str)

def recreate(out_str):
    # -----调试代码-----
    print("==========eval.evaluate: recreate starts==========")
    # ------------------
    kvs = parse_key_values(out_str)
    # -----调试代码-----
    print("kvs: ", kvs)
    print("==========eval.evaluate: recreate ends==========")
    # ------------------
    return {kv[0].replace("\\",""):kv[1] for kv in kvs}
    
def clean_output(id,out_str):
    # -----调试代码-----
    print("==========eval.evaluate: clean_output starts==========")
    print(f"id: {id}, out_str: {out_str}")
    # ------------------
    try:
        if np.isnan(out_str):
            import pdb;pdb.set_trace()
        out_str = out_str.strip().split("\n")[0]
        # -----调试代码-----
        print(f"id: {id}, out_str: {out_str}")
        # ------------------
        out_str = out_str.replace("Stop Here","")
        # -----调试代码-----
        print(f"id: {id}, out_str: {out_str}")
        # ------------------
        out_str = out_str.strip()
        # -----调试代码-----
        print(f"id: {id}, out_str: {out_str}")
        # ------------------
        out_str = out_str.replace("'s","s")
        # -----调试代码-----
        print(f"id: {id}, out_str: {out_str}")

        # ------------------
        #out_str = re.sub(r":\s*'",':"""',out_str)
        #out_str = re.sub(r"'\s*}",'"""}',out_str)
        #out_str = re.sub(r"'\s*,",'""",',out_str)
        out_str = escaped_(out_str)
        # -----调试代码-----
        print(f"id: {id}, out_str: {out_str}")
        print("==========eval.evaluate: clean_output ends==========")
        # ------------------
        return ast.literal_eval(out_str)
    except Exception as e:
        #{'cop'\s*:(.*),\s*['"]cop_index['"]:(.*),\s*['"]why_correct['"]:(.*),\s*['"]why_others_incorrect['"]:(.*)}
        b_str = out_str 
        out_str = recreate(out_str)
        if len(out_str.keys()) == 0:
            print(b_str)
        print("Exception during parsing data - recreated str",id,out_str,b_str)
        # -----调试代码-----
        print(f"id: {id}, out_str: {out_str}")
        print("==========eval.evaluate: clean_output ends==========")
        # ------------------
        return out_str
     
def convert_to_json(prediction_folder,dataset_folder):    
    
    pred_files = glob.glob(os.path.join(prediction_folder,"*.csv"))
    # -----调试代码-----
    print("==========eval.evaluate: convert_to_json starts==========")
    print("pred_files: ", pred_files)
    # ------------------
    for pred_file in pred_files:
        if os.path.basename(pred_file)=='results.csv':
            continue
        filename = os.path.basename(pred_file)
        # -----调试代码-----
        print("filename: ", filename)
        # ------------------
        prefix = filename.split(".")[0]
        # -----调试代码-----
        print("prefix: ", prefix)
        # ------------------
        dataset_name = pred_prefix_dict[prefix]
        # -----调试代码-----
        print("dataset_name: ", dataset_name)
        # ------------------
        dataset_df = pd.read_csv(os.path.join(dataset_folder,f"{dataset_name}.csv"),nrows=4)
        # -----调试代码-----
        print("dataset_df: ", dataset_df)
        # ------------------
        pred_df = pd.read_csv(pred_file,names=["id","output"])
        # -----调试代码-----
        print("pred_df: ", pred_df)
        # ------------------
        merge_df = pd.merge(left=dataset_df,right=pred_df,on=['id'])
        # -----调试代码-----
        print("merge_df after merge: ", merge_df)
        # ------------------
        merge_df["output"] = merge_df[['id','output']].fillna("").apply(lambda params : clean_output(*params),axis='columns')
        # -----调试代码-----
        print("merge_df after clean_output: ", merge_df)
        # ------------------
        merge_df = merge_df.replace({np.nan: None}) # 将DataFrame中的NaN值替换为None，以便在转换为JSON时正确处理
        merge_dict = merge_df.to_dict(orient='records')
        # -----调试代码-----
        print("merge_dict after to_dict: ", merge_dict)
        # ------------------
        
        print(f"Merging and converting the prediction to Json files - {dataset_name}")
        with open(os.path.join(prediction_folder,f"{dataset_name}.json"),'w') as fp:
            json.dump(merge_dict,fp)
    # -----调试代码-----
    print("==========eval.evaluate: convert_to_json ends==========")
    # ------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_folder",type=str, default="./medhalt/predictions/deepseek-r1-7b/")
    parser.add_argument("--dataset_folder",type=str, default="./medhalt/datasets/")
    parser.add_argument("--do_json_conversion",action='store_true')
    parser.add_argument("--point_score",action='store_true')
    
    args = parser.parse_args()
    # -----调试代码-----
    print("==========eval.evaluate: main starts==========")
    print("args: ", args)
    # ------------------

    results_df = pd.DataFrame()
    
    # if args.do_json_conversion:
    #     convert_to_json(args.prediction_folder,args.dataset_folder)
        
    for incorrect_score in [1,-0.25]:
        evaluator = FullDataEval(args.prediction_folder,1,incorrect_score)
        full_df = evaluator.run_all_evaluations()
        full_df["point_score"] = (incorrect_score==-0.25)
        results_df = pd.concat([full_df,results_df],ignore_index=True)
    
    results_df.to_csv(os.path.join(args.prediction_folder,"results.csv"),index=False)
    # -----调试代码-----
    print("==========eval.evaluate: main ends==========")
    # ------------------