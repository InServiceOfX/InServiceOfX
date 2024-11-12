def get_pad_token_id(model, tokenizer):
    if (model.config.pad_token_id is None):
        if (tokenizer.pad_token_id is None):
            return tokenizer.eos_token_id
        else:
            return tokenizer.pad_token_id
    else:
        return model.config.pad_token_id