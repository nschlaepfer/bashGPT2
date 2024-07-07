import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformer import GPT2, GPT2Config

# Step 1: Preparing the Dataset
dataset = load_dataset("csv", data_files={"train": "/Users/nicolas/Documents/GitHub/gpt2-mlx/BestData.csv"})

# Step 2: Tokenizing Bash Commands
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
special_tokens = ["ls", "cd", "mkdir", "rm", "touch", "-l", "-a", "--help"]
tokenizer.add_tokens(special_tokens)
tokenizer.pad_token = tokenizer.eos_token

# Step 3: Fine-Tuning GPT-2
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_layer=12,
    n_head=12,
    n_embd=768
)
model = GPT2(config)

# Function to tokenize dataset
def tokenize_function(examples):
    inputs = tokenizer(examples["Input"], padding="max_length", truncation=True, return_tensors="np")
    outputs = tokenizer(examples["Output"], padding="max_length", truncation=True, return_tensors="np")
    return {
        "input_ids": inputs["input_ids"],
        "labels": outputs["input_ids"]
    }

# Tokenize the dataset
tokenized_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Training loop
def train_step(model, batch, optimizer):
    def loss_fn(model):
        logits, _ = model(batch["input_ids"])
        shift_logits = logits[:, :-1, :]
        shift_labels = batch["labels"][:, 1:]
        loss = nn.losses.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1))
        return mx.mean(loss)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    optimizer.update(model, grads)
    return loss

# Training configuration
num_epochs = 5
batch_size = 8
learning_rate = 5e-5

# Initialize optimizer
optimizer = optim.Adam(learning_rate=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(tokenized_dataset), batch_size):
        batch = tokenized_dataset[i:i+batch_size]
        # Convert to MLX arrays here
        mlx_batch = {k: mx.array(v) for k, v in batch.items()}
        loss = train_step(model, mlx_batch, optimizer)
        
        if i % 100 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")
    
    # Optionally save the model after each epoch
    mx.savez(f"gpt2_mlx_epoch_{epoch+1}.npz", **dict(tree_flatten(model.parameters())))

print("Fine-tuning completed.")

# Save the final fine-tuned model
mx.savez("gpt2_mlx_final.npz", **dict(tree_flatten(model.parameters())))
print("Model saved as 'gpt2_mlx_final.npz'.")