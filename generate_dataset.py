from datasets import load_dataset, Dataset
import pandas as pd
import random
from tqdm import tqdm
import argparse


def generate_dataset(n_samples, n_shots, cut_fs, fs_max_length):
    dataset = load_dataset("Ego/jpflan-raw")["train"]
    templates = pd.read_csv("templates.csv")[["Dataset_Name", "template1", 
                                                  "template2", "template3", 
                                                  "fs_template", "fs"]]
    dataset_info = pd.read_csv("dataset_info.csv")

    train = {
        "dataset_name": [],
        "input": [],
        "output": [],
        "shots": []
    }

    tmp_ln = 0
    for dt_name in templates["Dataset_Name"]:
        ln = dataset_info[dataset_info["Dataset Name"] == dt_name]["Data Size"].values[0]

        # get the template row corresponding to the dataset
        template_row = templates[templates["Dataset_Name"] == dt_name]

        # shuffle the dataset
        tmp_dataset = dataset.select(list(range(tmp_ln, tmp_ln + ln)))
        tmp_dataset = tmp_dataset.shuffle(seed=0)

        # get the samples
        samples = min(int(n_samples), len(tmp_dataset))
        tmp_dataset = tmp_dataset.select(list(range(samples)))

        train["dataset_name"].extend([dt_name] * samples)
        train["shots"].extend([0] * samples)
        
        for sample in tqdm(tmp_dataset):
            prompt = random.choice(template_row[["template1", "template2", 
                                                 "template3"]].values.tolist()[0])

            inp_template = prompt.split("[/INST]")[0] + "[/INST]"
            out_template = prompt.split("[/INST]")[1]

            inputs = {}
            for k, v in sample.items():
                if "text" in k:
                    inputs[dataset_info[dataset_info["Dataset Name"] == dt_name][k].values[0]] = v

            train["input"].append(inp_template.format(**inputs))
            train["output"].append(out_template.format(**inputs))
        
        shots_list = []
        # sample ln examples from the dataset (few-shot)
        seed = 1
        tmp_dataset = tmp_dataset.shuffle(seed=7777777)
        if n_shots and dt_name not in ["syosetsu"]:
            for sample in tqdm(tmp_dataset):
                # get few-shot prompt
                prompt = template_row["fs"].values.tolist()[0]
                # add output template to the fs prompt
                fs_template = "\n" + template_row["fs_template"].values.tolist()[0] + "\n" + prompt.split("[/INST]")[1] + "\n"
                num_fs = random.randint(1, n_shots)
                # sample num_fs examples from the dataset
                tmp_few_shot_samples = tmp_dataset.shuffle(seed=seed).select(list(range(num_fs + 1)))
                    
                fs = ""
                shots = 0
                for fs_sample in tmp_few_shot_samples:
                    fs_input = {}
                    if sample != fs_sample:
                        for k, v in fs_sample.items():
                            if "text" in k:
                                fs_input[dataset_info[dataset_info["Dataset Name"] == dt_name][k].values[0]] = v

                        fs += fs_template.format(**fs_input)
                        shots += 1

                    if shots == num_fs:
                        break

                    if cut_fs and len(fs) > fs_max_length:
                        break

                shots_list.append(shots)

                inp_template = prompt.split("[/INST]")[0] + "[/INST]"
                out_template = prompt.split("[/INST]")[1]

                inputs = {}
                for k, v in sample.items():
                    if "text" in k:
                        inputs[dataset_info[dataset_info["Dataset Name"] == dt_name][k].values[0]] = v
                inputs["fs_template"] = fs

                train["input"].append(inp_template.format(**inputs))
                train["output"].append(out_template.format(**inputs))

                seed += 1

        train["shots"].extend(shots_list)
        train["dataset_name"].extend([dt_name] * len(shots_list))
        tmp_ln += ln    

    # Create and save the Huggingface dataset
    train_dataset = Dataset.from_dict(train)
    train_dataset.save_to_disk("./jpflan/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Japanese instruction tuning dataset for LLM training")
    parser.add_argument("--samples", default=10000, type=int, help="Number of samples to use")
    parser.add_argument("--n_shots", default=3, type=int, help="Number of shots to use for few-shot learning")
    parser.add_argument("--cut_fs", default=False, type=bool, help="Stop adding few-shot examples if the length exceeds fs_max_length")
    parser.add_argument("--fs_max_length", default=1000, type=int, help="Maximum length of the few-shot examples")
    args = parser.parse_args()

    print(f"Samples: {args.samples}, N-few-shot: {args.n_shots}, Cut-fs: {args.cut_fs}, Fs-max-length: {args.fs_max_length}")
    generate_dataset(args.samples, args.n_shots, args.cut_fs, args.fs_max_length)

