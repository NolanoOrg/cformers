"""Call's the C++ code from Python."""
from subprocess import Popen, PIPE
import hashlib
import os
import sys
import transformers as tf # RIP TensorFlow
import select

sys.path.append("./cpp/")

# Get CFORMERS_CACHE_PATH if it exists as an environment variable else use the default
if "CFORMERS_CACHE_PATH" in os.environ:
    CFORMERS_CACHE_PATH = os.environ["CFORMERS_CACHE_PATH"]
else:
    CFORMERS_CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cformers")

class ModelUrlMap:
    """Stores the URL mapping for various models."""
    def __init__(self,
                 cpp_model_name,
                 int4_fixed_zero="",
                 int4_variable_zero="",
                 gptq_int4_fixed_zero="",
                 gptq_int4_variable_zero=""):
        """Constructor for ModelUrlMap"""
        self.cpp_model_name = cpp_model_name
        self.int4_fixed_zero = int4_fixed_zero
        self.int4_variable_zero = int4_variable_zero
        self.gptq_int4_fixed_zero = gptq_int4_fixed_zero
        self.gptq_int4_variable_zero = gptq_int4_variable_zero

    def get_url(self, mode):
        """Returns the URL for the given mode."""
        url = None
        if mode == "int4_fixed_zero":
            url = self.int4_fixed_zero
        elif mode == "int4_variable_zero":
            url = self.int4_variable_zero
        elif mode == "gptq_int4_fixed_zero":
            url = self.gptq_int4_fixed_zero
        elif mode == "gptq_int4_variable_zero":
            url = self.gptq_int4_variable_zero
        else:
            raise ValueError("Invalid mode: {}".format(mode))

        if self.url == "":
            raise ValueError("{} not available for this model, please choose from: {}".format(mode, self.get_modes()))
        return url
        
    def get_modes(self):
        return [mode for mode in ["int4_fixed_zero", "int4_variable_zero", "gptq_int4_fixed_zero", "gptq_int4_variable_zero"] if self.get(mode) != ""]

MAP_MODEL_TO_URL = { # Replace "/" with "---" in the model name
    'EleutherAI/gpt-j-6B': ModelUrlMap(
        cpp_model_name="gptj",
        int4_fixed_zero="https://huggingface.co/ayushk4/EleutherAI---gpt-j-6B/resolve/main/int4_fixed_zero.bin"),
    'bigscience/bloom-7b1': ModelUrlMap(
        cpp_model_name="bloom",
        int4_fixed_zero="https://huggingface.co/ayushk4/bigscience---bloom-7b1/resolve/main/int4_fixed_zero.bin"),
}

class AutoInference:
    """A wrapper for the C++ model."""
    def __init__(self, model_name, hash_sum, mode="int4_fixed_zero"):
        self.model_name = model_name
        self.mode = mode
        self.hash_sum = hash_sum
        self.cpp_model_name = MAP_MODEL_TO_URL[model_name].cpp_model_name
        self.model_url = MAP_MODEL_TO_URL[model_name].get_url(mode)
        self.model_save_path = os.path.join(CFORMERS_CACHE_PATH, "models", model_name, mode)
        self.tokenizer = tf.AutoTokenizer.from_pretrained(model_name)

        # Download the model if it doesn't exist
        if not os.path.exists(self.model_save_path):
            print("Downloading model...")
            tf.file_utils.cached_path(self.model_url, cache_dir=self.model_save_path)

        # TODO: Check the hash sum of the downloaded model

    def generate(self,
                 prompt,
                 top_k=20,
                 top_p=0.95,
                 temperature=00.85,
                 num_tokens_to_generate=10,
                 repeat_last_n=64,
                 repeat_penalty=1.3,
                 n_threads=8,
                 seed=42,
                 print_streaming_output=True):
        """Generates text from the given prompt."""
        if isinstance(prompt, str):
            # Tokenize and get the input ids
            prompt = self.tokenizer.encode(prompt)['input_ids']
        # By now prompt should be a list of integers, sanity check this once
        assert isinstance(prompt, list), f"Prompt should be a list of integers: {prompt}"
        assert all([isinstance(x, int) for x in prompt]), \
            f"Prompt should be a list of integers {prompt}"
        # Convert to a string of space separated integers
        prompt = " ".join([str(x) for x in prompt])

        process = Popen(["./cpp/main", self.cpp_model_name,
                         "-m", self.cpp_model_name,
                         "--prompt", prompt,
                         "--seed", str(seed),
                         "--threads", str(n_threads),
                         "--n_predict", str(num_tokens_to_generate),
                         "--top_k", str(top_k),
                         "--top_p", str(top_p),
                         "--temp", str(temperature),
                         "--repeat_last_n", str(repeat_last_n),
                         "--repeat_penalty", str(repeat_penalty)],
                        stdout=PIPE, stderr=PIPE)

        while True:
            # Wait for data to be available on the process's standard output stream
            select.select([process.stdout.fileno()], [], [])

            # Read available data from the stream and print it to the console
            output = process.stdout.readline().decode('utf-8')
            if output == '' and process.poll() is not None:
                break
            print(output.strip())

        # Wait for the process to finish and return its exit code
        return process.wait()
