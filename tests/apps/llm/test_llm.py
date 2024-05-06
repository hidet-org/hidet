import pytest
import asyncio
import random

from hidet.apps.llm import create_llm
from hidet.apps.llm.sampler import SamplingParams

hf_model = "meta-llama/Llama-2-7b-chat-hf"
samples = {
    'Hello, how are you?': "I'm doing well, thanks for asking! *smiles* It'",
    'How do you feel about the current political climate?': "\n\nI'm just an AI, I don't have personal",
    'What is your favorite food?': '\n\nMy favorite food is pizza. I love the combination of the cr',
    'What is your favorite color?': "\n\nI'm just an AI, I don't have personal",
    'What is your favorite animal?': '\n\nI love all animals, but if I had to choose just one,',
}


async def _demo_async():
    llm = create_llm(hf_model)
    coros = []
    for prompt, expected_output in samples.items():

        async def f(prompt: str, expected_output: str):
            await asyncio.sleep(random.randint(1, 5))
            print("Incoming request: ", prompt)
            params = SamplingParams(temperature=0.0, max_tokens=16)
            stream = llm.async_generate(prompt, sampling_params=params)
            final = None
            async for output in stream:
                # print(output.tokens)
                final = output
            actual_output = final.output_text
            print("Request finished: {} \nOutput: {}".format(prompt, actual_output))
            if actual_output != expected_output:
                raise ValueError(
                    "Prompt: {}\nExpected: {}\nActual: {}\n".format(
                        repr(prompt), repr(expected_output), repr(actual_output)
                    )
                    + "Output does not match expected output."
                )
            assert final.output_text == expected_output

        coros.append(f(prompt, expected_output))

    await asyncio.gather(*coros)


@pytest.mark.release
def test_async():
    asyncio.run(_demo_async())


@pytest.mark.release
def test_sync():
    llm = create_llm(hf_model)
    prompts = [sample[0] for sample in samples.items()]
    actual_outputs = [
        output.output_text
        for output in llm.generate(prompts, sampling_params=SamplingParams(temperature=0.0, max_tokens=16))
    ]
    expected_outputs = [sample[1] for sample in samples.items()]
    for prompt, actual_output, expected_output in zip(prompts, actual_outputs, expected_outputs):
        if actual_output != expected_output:
            raise ValueError(
                "Prompt: {}\nExpected: {}\nActual: {}\n".format(
                    repr(prompt), repr(expected_output), repr(actual_output)
                )
                + "Output does not match expected output."
            )


if __name__ == "__main__":
    pytest.main([__file__])
