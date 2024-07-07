from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Set the model name
model_name = "gpt2"

# Set the root directory for saving the model
root_dir = os.getcwd()

# Download and save the model
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=False, cache_dir=root_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False, cache_dir=root_dir)

# Save the model and tokenizer
model.save_pretrained(os.path.join(root_dir, model_name))
tokenizer.save_pretrained(os.path.join(root_dir, model_name))

print(f"Model and tokenizer saved to {os.path.join(root_dir, model_name)}")

# You can now use the model as before
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model=model_name)
set_seed(42)
output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
print(output)