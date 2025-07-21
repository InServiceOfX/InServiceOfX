def get_tokens_per_second_statistics(
        input_token_count,
        output_token_count,
        start_time,
        end_time):
    generated_token_count = output_token_count - input_token_count
    total_time = end_time - start_time
    stats = {
        'total_time_seconds': total_time,
        'input_token_count': input_token_count,
        'output_token_count': output_token_count,
        'generated_token_count': generated_token_count,
        'input_tokens_per_second': \
            input_token_count / total_time if total_time > 0 else 0,
        'generated_tokens_per_second': \
            generated_token_count / total_time if total_time > 0 else 0,
        'output_tokens_per_second': \
            output_token_count / total_time if total_time > 0 else 0,
    }

    return stats
