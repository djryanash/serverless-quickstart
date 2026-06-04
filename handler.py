import runpod
import time
import os
from datetime import datetime
from vllm import LLM, SamplingParams

# Mock vLLM if not on Linux/GPU
if os.name != 'nt' and not os.path.exists('/usr/local/cuda'):
    class LLM:
        def __init__(self, **kwargs): pass
        def generate(self, prompts, **kwargs):
            class MockOutput:
                class MockOutputItem:
                    text = "Mocked response"
                outputs = [MockOutputItem()]
            return [MockOutput()]
    class SamplingParams:
        def __init__(self, **kwargs): pass
else:
    from vllm import LLM, SamplingParams

# Load model once at startup
# Using Qwen2.5-1.8B for faster cold starts
llm = LLM(model="Qwen/Qwen2.5-1.8B", tensor_parallel_size=1)


def handler(event):
    """
    Process incoming requests to your Serverless endpoint.

    Args:
        event (dict): Contains the input data and request metadata

    Returns:
        dict: Structured response with output, timing, and metadata
    """
    input_data = event.get('input', {})
    prompt = input_data.get('prompt')

    # Input validation
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        return {
            "output": None,
            "error": "Missing or empty 'prompt' field",
            "status": 400
        }

    sampling_params = SamplingParams(temperature=0.7)

    start_time = time.time()
    print(f"Received prompt: {prompt}")

    try:
        # Generate response using vLLM
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "output": generated_text,
            "timing": {
                "elapsed_ms": elapsed_ms
            },
            "model": "Qwen/Qwen2.5-1.8B",
            "status": 200
        }

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "output": None,
            "error": str(e),
            "timing": {
                "elapsed_ms": elapsed_ms
            },
            "status": 500
        }


# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
