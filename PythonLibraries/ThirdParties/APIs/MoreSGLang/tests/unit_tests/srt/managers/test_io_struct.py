from sglang.srt.managers.io_struct import GenerateReqInput

def test_generate_req_input():

    prompts = [
        "Who is the president of the United States currently?",
        "What is the capital of France?",
        "What is the future of AI?"
    ]
    # text
    # The input prompt. It can be a single prompt or a batch of prompts.

    # return_logprob
    # Whether to return logprobs. bool.

    # logprob_start_len
    # If return_logprobs, start location in prompt for returning lobprobs.
    # By default, this value is "-1", which means it'll only return logprobs for
    # output tokens.

    object = GenerateReqInput(prompts)