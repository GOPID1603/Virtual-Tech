import os
# Suppress TensorFlow/PyTorch warnings if any
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline, set_seed
import time

def main():
    print("=== Virtual Technologies Internship: AI Task 5 - Generative AI Text Model ===")
    print("Initializing HuggingFace Transformers Pipeline...")
    print("(Note: This may take a moment to download the model the very first time it runs)\n")
    
    # Initialize the text-generation pipeline using a fast, small model (GPT-2)
    # Using a smaller model ensures it runs quickly without a massive download.
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42) # For reproducible results
    
    # Pre-defined prompts to test the Generative AI
    prompts = [
        "The future of Artificial Intelligence in healthcare is",
        "Once upon a time in a cyberpunk city,",
        "def fibonacci(n):"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: '{prompt}'")
        print("Generating response...")
        
        start_time = time.time()
        
        # Generate text
        response = generator(prompt, max_length=50, num_return_sequences=1, truncation=True, pad_token_id=50256)
        
        end_time = time.time()
        
        generated_text = response[0]['generated_text']
        
        print("\n[AI Output]:")
        print(f"\033[92m{generated_text}\033[0m") # Print in green if terminal supports it
        print(f"(Generated in {end_time - start_time:.2f} seconds)")
        time.sleep(1)
        
    print("\n=== AI Task 5 Completed Successfully ===")

if __name__ == "__main__":
    main()
