import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import pandas as pd
import collections
import json
import pickle


class VideoQA_Dataset(Dataset):
    def __init__(self, args, tokenizer, split):
        
        path = f'./meta_data/{args.dataset}/'
        self.data = pd.read_csv(path + f'{split}.csv')
        self.features = torch.load(path + f'clipvitl14.pth')
        self.ans2id = json.load(open(path + f'{split}_vocab.json'))
        self.ans2cat = json.load(open(path + 'ans2cat.json'))
        self.max_feats = args.max_feats
        self.features_dim = args.features_dim
        self.split = split
        self.prefix = args.prefix
        self.suffix = args.suffix
        self.mask = tokenizer.mask_token
        self.pad = tokenizer.pad_token
        self.tokenizer = tokenizer
        self.use_context = (args.use_context and args.dataset != "tgif")
        self.subs = pickle.load(open(path + f'subtitles.pkl', "rb")) if self.use_context else None
        self.args = args
        self.load_answer_graph()

    def load_answer_graph(self):
        self.edge_index = torch.load(f'./meta_data/{self.args.dataset}/answer_graph/{self.split}_edge_index.pth')
        self.vocab_embeddings = torch.load(f'./meta_data/{self.args.dataset}/answer_graph/{self.split}_x.pth') 
        cat2coef = {'base': 1.0, 'common': self.args.eps, 'rare': self.args.eps, 'unseen': self.args.eps}
        # cat2coef = {'base': self.args.eps, 'common': self.args.eps, 'rare': self.args.eps, 'unseen': self.args.eps}
        self.eps = torch.tensor([cat2coef[self.ans2cat[k]] for k, v in self.ans2id.items()])
    
    def __len__(self):
        return len(self.data)

    def _get_text(self, question, mask, sub):
        text = f"{self.prefix} Question: {question} Answer: {mask}{self.suffix}"
        if sub:
            text += f" Subtitles: {sub}"
        text = text.strip()
        return text

    def _get_video(self, video_id):
        if video_id not in self.features:
            print(video_id)
            video = torch.zeros(1, self.features_dim)
        else:
            video = self.features[video_id].float()
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], 0)
        else:
            video_len = self.max_feats

        return video, video_len

    def __getitem__(self, idx):
        # get question
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
        type = 0
        if "type" in self.data:
            type = self.data["type"].values[idx]

        
        original_answer = self.data["answer"].values[idx]
        answer_id = self.ans2id[original_answer]
        video_id = self.data["video_id"].values[idx]

        # get subtitles
        sub = ""
        if self.subs is not None and video_id in self.subs:
            sub = self.subs[video_id]
        sub_bool = bool(sub)
        if not self.use_context:
            sub = ""

        # get pattern
        text = self._get_text(question, self.mask, sub)

        # get video
        video, video_len = self._get_video(video_id)

        return {"video": video, "video_len": video_len, "text": text, "qid": idx, "answer_id": answer_id,
                "type": type, "sub": sub_bool, "original_answer": original_answer}


def videoqa_collate_fn(batch):
    bs = len(batch)
    video = torch.stack([batch[i]["video"] for i in range(bs)])
    video_len = torch.tensor([batch[i]["video_len"] for i in range(bs)], dtype=torch.long)
    text = [batch[i]["text"] for i in range(bs)] if isinstance(batch[0]["text"], str) else [[batch[i]["text"][j] for i in range(bs)] for j in range(len(batch[0]["text"]))]
    qid = [batch[i]["qid"] for i in range(bs)]
    answer_id = default_collate([batch[i]["answer_id"] for i in range(bs)])
    type = [batch[i]["type"] for i in range(bs)]
    original_answer = [batch[i]["original_answer"] for i in range(bs)]
    out = {"video": video, "video_len": video_len, "text": text, "qid": qid, "answer_id": answer_id, "type": type, "original_answer": original_answer}
    if "sub" in batch[0]:
        sub = [batch[i]["sub"] for i in range(bs)]
        out["sub"] = sub
    return out