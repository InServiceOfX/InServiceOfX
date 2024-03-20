with gr.Blocks(theme=Base(), title="Question Answering App using Vector Search + RAG") as demo:
    gr.Markdown(
        """
        # Question Answering App using Atlas Vector Search + RAG Architecture
        """)
    textbox = gr.Textbox(label="Enter your Question:")
    with gr.Row():
        button = gr.Button("Submit", variant="primary")
    with gr.Column():
        output1 = gr.Textbox(lines=1, max_lines=10, label="Output with just Atlas Vector Search (returns text field as is):")
        output2 = gr.Textbox(lines=1, max_lines=10, label="Output generated by chaining Atlas Vector Search to Langchain's RetrieverQA + OpenAI LLM:")

# Call query_data function upon clicking the Submit button

    button.click(query_data, textbox, outputs=[output1, output2])


if __name__ == '__main__':
    demo.launch()