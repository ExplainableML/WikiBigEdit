import math
import sys

import torch
from torch.nn.functional import cosine_similarity
from transformers import PreTrainedModel, PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from annoy import AnnoyIndex
from tqdm import tqdm
from time import time


class RAG(torch.nn.Module):
    def __init__(self, config, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: str):
        """
        Initializes the RAG model.
        Args:
            config: Configuration for the RAG model.
            model: The language model (e.g., from HuggingFace).
            tokenizer: The tokenizer corresponding to the language model.
            device: The device for model computation ('cpu' or 'cuda').
        """
        super(RAG, self).__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.memory = []  # Initialize an empty list for memory storage.
        self.top_k = config.top_k
        self.exact_match = config.exact_match
        self.solver = config.solver
        self.solver_args = config.solver_args

        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

        self.index_set = False
        self.index = None

        self.verbose = False
        self.out = []

        if not self.exact_match and self.solver_args.gpu:
            self.res = faiss.StandardGpuResources()
        else:
            self.res = None

    def _retrieve_top_k(self, input_embeddings, qs):
        """
        Retrieves the top-k closest elements from memory using cosine similarity.
        Args:
            input_embeddings: The embedding of the current input.
        Returns:
            A list of top-k closest elements from memory.
        """
        if not self.memory:
            return []

        if len(self.memory) < self.top_k:
            return [f'Q: {item["question"]} A: {item["answer"]}' for item in self.memory]

        memory_embeddings = torch.stack([item['embedding'] for item in self.memory]).to(self.device)
        memory_embeddings = memory_embeddings.squeeze(1)
        input_embeddings = torch.nn.functional.normalize(input_embeddings, p=2, dim=1)
        similarities = torch.matmul(input_embeddings, memory_embeddings.T)

        top_k_indices = torch.topk(similarities, self.top_k, dim=-1).indices
        top_k_items = []
        for b in range(top_k_indices.shape[0]):
            top_k_items.append([f'Q: {self.memory[i]["question"]} A: {self.memory[i]["answer"]}' for i in top_k_indices[b].tolist()])

        return top_k_items

    def set_index_old(self):
        # Prepare memory embeddings as a NumPy array
        memory_embeddings_np = np.stack([item['embedding'].cpu().numpy() for item in self.memory])
        embedding_dim = memory_embeddings_np.shape[1]

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(memory_embeddings_np)

        # Create a Product Quantizer index
        # Use `d` dimensions, 8 subquantizers
        n_subquantizers = 8  # Number of subquantizers (tradeoff between speed and accuracy)
        n_bits = 8  # Bits per code
        self.index = faiss.IndexPQ(embedding_dim, n_subquantizers, n_bits)

        # Train the index with memory embeddings
        self.index.train(memory_embeddings_np)  # PQ requires training
        self.index.add(memory_embeddings_np)  # Add embeddings to the index
        self.index_set = True

    def set_index(self, nlist=None):
        if self.solver == 'annoy_HNSW':
            embedding_dim = self.memory[0]['embedding'].shape[0]
            print(f'Embedding dim: {embedding_dim}')
            self.index = AnnoyIndex(embedding_dim, self.solver_args.metric)
            print(f'Index initialized')
            for idx, item in enumerate(self.memory):
                self.index.add_item(idx, item['embedding'])
            print(f'Items added')
            self.index.build(self.solver_args.n_trees)
            print(f'Index built')
        else:
            memory_embeddings_np = np.stack([item['embedding'].cpu().numpy() for item in self.memory])
            embedding_dim = memory_embeddings_np.shape[1]
            # Normalize embeddings
            faiss.normalize_L2(memory_embeddings_np)
            quantizer = faiss.IndexFlatIP(embedding_dim)  # Coarse quantizer
            if self.solver == 'FlatIP':
                self.index = quantizer
            elif self.solver == 'IVFPQ':
                # Number of clusters for IVF
                n_subquantizers = max(embedding_dim // 4, self.solver_args.n_subquantizers)
                if nlist is None:
                    nlist = int(math.sqrt(len(memory_embeddings_np)))
                self.index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, n_subquantizers, self.solver_args.n_bits)
                self.index.nprobe = self.solver_args.nprobe  # Number of clusters to search
            elif self.solver == 'HNSW':
                self.index = faiss.IndexHNSWFlat(embedding_dim, self.solver_args.hnsw_m)
                self.index.hnsw.efConstruction = self.solver_args.hnsw_efConstruction
                self.index.hnsw.efSearch = self.solver_args.hnsw_efSearch
            else:
                raise ValueError(f"Solver {self.solver} is not supported.")

            if self.solver_args.gpu:
                self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)

            # Train and add embeddings
            if self.solver == 'IVFPQ':
                if len(memory_embeddings_np) < 10 * nlist:
                    raise ValueError("Not enough embeddings to train the IVF-PQ index.")
                self.index.train(memory_embeddings_np)
            self.index.add(memory_embeddings_np)

        self.index_set = True

    def _retrieve_top_k_faiss(self, input_embeddings):
        """
        Retrieves the top-k closest elements from memory using Faiss with Product Quantization (PQ).
        Args:
            input_embeddings: The embedding of the current input (Tensor of shape [batch_size, embedding_dim]).
        Returns:
            A list of top-k closest elements from memory.
        """
        if not self.memory:
            return []

        if len(self.memory) < self.top_k:
            return [item['text'] for item in self.memory]

        if not self.index_set:
            self.set_index()

        if self.solver == 'annoy_HNSW':
            indices = [self.index.get_nns_by_vector(emb, self.top_k, include_distances=False) for emb in input_embeddings]
        else:
            input_embeddings_np = input_embeddings.cpu().numpy()
            faiss.normalize_L2(input_embeddings_np)
            distances, indices = self.index.search(input_embeddings_np, self.top_k)
        # Retrieve top-k items from memory
        top_k_items = []
        for i in range(len(indices)):
            top_k_items.append([f'Q: {self.memory[i]["question"]} A: {self.memory[i]["answer"]}' for i in indices[i]])
        return top_k_items

    def to_memory(self, batch):
        q = [item['prompt'] for item in batch]
        a = [item['target_new'] for item in batch]

        q_embeddings = self.embedding_model.encode(q, show_progress_bar=False)
        """
        tokenized = self.tokenizer(
            q,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)

        with torch.no_grad():
            input_embeddings = self.model.get_input_embeddings()(input_ids)
            sum_embeddings = torch.sum(input_embeddings * attention_mask.unsqueeze(-1), dim=1)
            token_counts = torch.sum(attention_mask, dim=1, keepdim=True)
            token_counts = torch.clamp(token_counts, min=1)
            mean_embeddings = sum_embeddings / token_counts

        # Normalize the embeddings
        input_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
        """
        for input_embedding, q, a in zip(q_embeddings, q, a):
            self.memory.append({'question': q, 'answer': a, 'embedding': input_embedding})

        #if not self.exact_match and self.index_set:
        #    self.index.add(input_embeddings.cpu().numpy())
        self.index_set = False

    def adapt(self, batched_requests):
        exec_times = []
        for i, request in enumerate(tqdm(batched_requests)):
            start = time()
            self.to_memory(request)

            exec_times.append(time() - start)
        return exec_times

    def forward(self, input_ids, attention_mask):
        """
        Performs a forward pass for the RAG model.
        Args:
            input_ids: Tokenized input IDs.
            attention_mask: Attention mask for the input IDs.
        Returns:
            The output of the language model.
        """
        if len(self.memory) > 0:
            len_input = input_ids.shape[1]
            org_padding_lengths = (input_ids == self.tokenizer.pad_token_id).sum(dim=1)
            original_inputs = [self.tokenizer.decode(input_ids[b], skip_special_tokens=True) for b in
                               range(input_ids.shape[0])]
            org_prompt = [original_inputs.split('? ')[0][3:] + '?' for original_inputs in original_inputs]
            input_embedding = self.embedding_model.encode(org_prompt, show_progress_bar=False)
            """
            #print(org_prompt)
            org_prompt_tok = self.tokenizer(org_prompt, return_tensors='pt', padding=True)
            org_prompt_ids = org_prompt_tok['input_ids'].to(self.device)
            attention_mask = org_prompt_tok['attention_mask'].to(self.device)
            # Compute the embedding of the input using the language model.
            with torch.no_grad():
                input_embeddings = self.model.get_input_embeddings()(org_prompt_ids)
                sum_embeddings = torch.sum(input_embeddings * attention_mask.unsqueeze(-1), dim=1)
                token_counts = torch.sum(attention_mask, dim=1, keepdim=True)
                token_counts = torch.clamp(token_counts, min=1)
                input_embedding = sum_embeddings / token_counts
            """
            # Retrieve top-k closest elements during eval mode.
            if self.exact_match:
                retrieved_contexts = self._retrieve_top_k(input_embedding, org_prompt)
            else:
                retrieved_contexts = self._retrieve_top_k_faiss(input_embedding)
            contexts = [" ".join(retrieved_context) for retrieved_context in retrieved_contexts]

            new_input = [f"{context} {original_input}".strip() for context, original_input in zip(contexts, original_inputs)]

            #print(f'Original input: {original_inputs}')
            #print(f'New input: {new_input}')

            # Tokenize the new input.
            new_input_ids = self.tokenizer(new_input, return_tensors='pt', padding=True, truncation=True, max_length=self.config.max_length)
            #print(f'Input ids: {input_ids}')
            #print(f'New input ids: {new_input_ids["input_ids"]}')
            #sys.exit()
            input_ids = new_input_ids['input_ids'].to(self.device)
            attention_mask = new_input_ids['attention_mask'].to(self.device)

        try:
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        except torch.cuda.OutOfMemoryError as e:
            print(f'Len input_ids: {input_ids.shape}')
            raise e

        if len(self.memory) > 0:
            if self.verbose:
                text_answers = [self.tokenizer.decode(ans, skip_special_tokens=True) for ans in
                                torch.argmax(out.logits, dim=-1).squeeze().detach().cpu().numpy().tolist()]
                for context, original_input, answ in zip(contexts, original_inputs, text_answers):
                    self.out.append([context, original_input, answ])
            # Remove the context from the output.
            len_context = input_ids.shape[1] - len_input
            # Check if padding left
            right_pad = [l_f[-1] == self.tokenizer.pad_token_id for l_f in input_ids]
            if any(right_pad):
                logits = out.logits.clone()
                logits_processed = []
                padding_lengths = (input_ids == self.tokenizer.pad_token_id).sum(dim=1)
                for b in range(out.logits.shape[0]):
                    if right_pad[b]:
                        padding_diff = padding_lengths[b] - org_padding_lengths[b]
                        if padding_diff <= 0:
                            logits_processed.append(logits[b, len_context:])
                        else:
                            logits_processed.append(logits[b, len_context-padding_diff:-padding_diff])
                    else:
                        logits_processed.append(logits[b, len_context:])
                out.logits = torch.stack(logits_processed)
            else:
                out.logits = out.logits[:, len_context:]

            #print(f'Right pad: {right_pad}')
            #print(f'Padding lengths: {padding_lengths}')
            #print(f'Input ids: {input_ids}')
            #print(f'Len context: {len_context}')
            #print(f'Len input: {len_input}')
            #print(f'Logit shape: {out.logits.shape}')
            #answers_full = torch.argmax(out.logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
            #print(f'Answers: {answers_full}')
            #text_answers = [self.tokenizer.decode(ans, skip_special_tokens=True) for ans in answers_full]
            #print(f'Answers: {text_answers}')
            #sys.exit()

        return out

