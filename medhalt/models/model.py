import os
import json
import time
from tqdm import tqdm
from torch.utils.data import DataLoader  # 仅保留 DataLoader 用于数据集批处理（可替换为纯 pandas 批处理）
from functools import partial
import asyncio
import csv
import ollama  # Ollama 核心库
import requests  # 用于 Ollama HTTP API 调用（可选）
import aiohttp  # 异步 HTTP 请求库（用于 Ollama HTTP API）

class Model:
    def __init__(self, ollama_base_url, model_name) -> None:
        """
        初始化 Ollama 模型
        :param ollama_base_url: Ollama HTTP API 地址（如 http://localhost:11434）
        :param model_name: Ollama 已拉取的模型名（如 deepseek-r1:7b）
        """
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name

    async def ollama_http_generate(self, prompt, **gen_kwargs):
        """使用 Ollama HTTP API 调用模型"""
        try:
            url = f"{self.ollama_base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": gen_kwargs.get("temperature", 0.6),
                    "max_tokens": gen_kwargs.get("max_new_tokens", 128),
                    "top_p": gen_kwargs.get("top_p", 0.95),
                }
            }
            print("http request payload: ", payload)
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data['response']
                    else:
                        return f"error:HTTP {resp.status} - {await resp.text()}"
        except Exception as e:
            return f"error:{str(e)}"

    # async def generate(self, prompt, **gen_kwargs):
    #     """统一推理入口：根据配置选择本地/HTTP API"""
    #     if self.use_http_api:
    #         return await self.ollama_http_generate(prompt, **gen_kwargs)
    #     else:
    #         return await self.ollama_local_generate(prompt, **gen_kwargs)

    async def batch_generate(self, batch_prompts, ids, **gen_kwargs):
        """批处理推理（异步）"""
        # -----调试代码-----
        print("==========models.model: batch_generate starts==========")
        # -----------------
        tasks = [self.ollama_http_generate(prompt, **gen_kwargs) for prompt in batch_prompts]
        results = await asyncio.gather(*tasks)

        # -----调试代码-----
        print("results after ollama_http_generate: ", results)
        print("==========models.model: batch_generate ends==========")
        # -----------------

        return results, ids

    def run_generation(self, dataset_name, prompt_template_fn, batch_size=16, output_folder=None, **gen_kwargs):
        """
        完整推理流程：加载数据集 → 批处理推理 → 保存结果
        """
        from medhalt.models.utils import PromptDataset

        outputs = []
        dataset = PromptDataset(dataset_name, prompt_template_fn)
        dataloader = DataLoader(dataset, batch_size, collate_fn=dataset._restclient_collate_fn)
        
        pred_folder = os.path.join(output_folder, self.model_name.replace(":", "-"))
        # -----调试代码-----
        print("==========models.model: run_generation starts==========")
        print(f"Saving predictions to: {pred_folder}")
        # ------------------
        os.makedirs(pred_folder, exist_ok=True)

        for batch in tqdm(dataloader):
            batch_prompts, ids = batch
            try:
                generated_texts, ids = asyncio.run(self.batch_generate(batch_prompts, ids, **gen_kwargs))
            except Exception as e:
                generated_texts = [f"error:{str(e)}"] * len(batch_prompts)
                ids = ["error"] * len(batch_prompts)

            with open(os.path.join(pred_folder, f"{dataset_name}.csv"), 'a', encoding='utf-8') as f: # 兼容Windows文件系统，强制指定编码方式为utf-8
                writer = csv.writer(f)
                for gtext, _id in zip(generated_texts, ids):
                    writer.writerow([_id, gtext]) 

            outputs.append({"generated_text": generated_texts, "id": ids})

        with open(os.path.join(pred_folder, "gen_kwargs.json"), 'w') as fp:
            json.dump(gen_kwargs, fp)

        # -----调试代码-----
        print("==========models.model: run_generation ends==========")
        # ------------------

        return outputs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ollama_base_url", type=str, default="http://localhost:11434")
    parser.add_argument("--model_name", type=str, default="deepseek-r1:7b")
    parser.add_argument("--dataset_name", type=str, default="all")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output_folder", type=str, default="./medhalt/predictions/")

    args = parser.parse_args()
    # -----调试代码-----
    print("==========models.model: main starts==========")
    print("args: ", args)
    # ------------------

    model_cls = Model(
        ollama_base_url=args.ollama_base_url,
        model_name=args.model_name
    )

    prompt_template_fn = lambda row: row

    for ds_name in ["Nota","fake", "FCT","abs2pub", "pmid2title", "url2title", "title2pub"]:
    # for ds_name in ["title2pub"]:
        try:
            print(f"Running predictions for - {ds_name}")
            generations = model_cls.run_generation(
                dataset_name=ds_name,
                prompt_template_fn=prompt_template_fn,
                batch_size=args.batch_size,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                output_folder=args.output_folder,
                stop_sequences=["Stop Here"],
                seed=42
            )
        except Exception as e:
            print(f"Error on {ds_name}: {e}")


    # -----调试代码-----
    print("==========models.model: main ends==========")
    # ------------------
