import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from time import time


class PromptTuning(torch.nn.Module):
    def __init__(self, config, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: str):
        """
        Initializes the Prompt Tuning model.
        Args:
            config: Configuration for the prompt tuning model.
            model: The base language model (e.g., from HuggingFace).
            tokenizer: The tokenizer corresponding to the language model.
            device: The device for model computation ('cpu' or 'cuda').
        """
        super(PromptTuning, self).__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Initialize learnable prompt token embeddings
        self.prompt_length = config.prompt_length
        self.token_dim = self.model.config.hidden_size
        self.prompt_embeddings = torch.nn.Parameter(
            torch.randn(self.prompt_length, self.token_dim, requires_grad=True)
        )
        self.to(self.device)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for the Prompt Tuning model.
        Args:
            input_ids: Tokenized input IDs.
            attention_mask: Attention mask for the input IDs.
            labels: Optional; labels for supervised learning tasks.
        Returns:
            The output of the language model.
        """
        # Get the input embeddings
        input_embeddings = self.model.get_input_embeddings()(input_ids)

        # Expand prompt embeddings to batch size and prepend to input embeddings
        batch_size = input_embeddings.size(0)
        prompt_embeddings = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        extended_embeddings = torch.cat([prompt_embeddings, input_embeddings], dim=1)

        # Extend attention mask for the prompt tokens
        extended_attention_mask = torch.cat(
            [torch.ones(batch_size, self.prompt_length, device=self.device), attention_mask], dim=1
        )

        # Pass through the model
        outputs = self.model(
            inputs_embeds=extended_embeddings,
            attention_mask=extended_attention_mask,
            labels=labels
        )
        outputs.logits = outputs.logits[:, self.prompt_length:]
        return outputs

    def adapt(self, batched_requests):
        start = time()
        self.train_prompt(batched_requests)
        exec_times = [time() - start]*len(batched_requests)
        return exec_times

    def train_prompt(self, batches):
        """
        Trains the prompt embeddings using question-answer pairs.
        Args:
            batches: List of batches containing question-answer pairs.
        """
        # ensure that only self.prompt_embeddings requires grad and self.model does not
        self.prompt_embeddings.requires_grad = True
        for param in self.model.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW([self.prompt_embeddings], lr=self.config.lr)

        # get the number of tunable parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of tunable parameters: {n_params}")

        for epoch in range(self.config.epochs):
            batch_losses = []
            for batch in batches:
                batch_questions = [item['prompt'] for item in batch]
                batch_answers = [item['target_new'] for item in batch]
                # Tokenize questions and answers
                inputs = self.tokenizer(
                    list(batch_questions),
                    list(batch_answers),
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    add_special_tokens=True
                ).to(self.device)

                input_ids = inputs["input_ids"]
                batch_size, input_seq_len = input_ids.shape

                # Prepend learnable prompt embeddings to input embeddings
                prompt_embeddings = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
                input_embeddings = self.model.get_input_embeddings()(input_ids)
                extended_embeddings = torch.cat([prompt_embeddings, input_embeddings], dim=1)

                # Prepare labels
                # Extend labels to match input length and prepend -100 for prompt tokens
                labels = input_ids.clone()  # Clone to avoid modifying input_ids
                label_padding = torch.full((batch_size, self.prompt_length), -100, device=self.device)
                extended_labels = torch.cat([label_padding, labels], dim=1)

                # Shift labels left by one position for autoregressive training
                shifted_labels = extended_labels.clone()  # Clone the tensor to avoid memory conflicts
                extended_labels[:, :-1] = shifted_labels[:, 1:]
                extended_labels[:, -1] = -100  # Ignore the last position as it has no prediction target

                # Extend attention mask for prompt tokens
                attention_mask = inputs["attention_mask"]
                extended_attention_mask = torch.cat(
                    [torch.ones(batch_size, self.prompt_length, device=self.device), attention_mask], dim=1
                )

                # Forward pass
                outputs = self.model(
                    inputs_embeds=extended_embeddings,
                    attention_mask=extended_attention_mask,
                    labels=extended_labels
                )
                loss = outputs.loss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            print(f"Epoch {epoch + 1}, Loss: {sum(batch_losses) / len(batch_losses)}")

    def inference(self, inputs):
        """
        Performs inference by prepending the learnable prompt token to inputs.
        Args:
            inputs: List of input texts.
        Returns:
            The generated outputs.
        """
        self.eval()
        with torch.no_grad():
            tokenized_inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(self.device)
            outputs = self.forward(input_ids=tokenized_inputs['input_ids'], attention_mask=tokenized_inputs['attention_mask'])
            predictions = torch.argmax(outputs.logits, dim=-1)
            return [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]