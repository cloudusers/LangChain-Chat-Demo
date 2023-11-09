import gradio as gr
from configs import *
from chains import ChatWithKBAndHistoryChain

llm_list = LLMS
embedding_list = EMBEDDINGS
default_llm = DEFAULT_LLM
default_embedding = DEFAULT_EMBEDDING
default_temperature = DEFAULT_TEMPERATURE
default_top_k = DEFAULT_TOP_K
chain = ChatWithKBAndHistoryChain(model=default_llm,
                                  embedding=default_embedding,
                                  temperature=default_temperature,
                                  top_k=default_top_k,
                                  with_knowledgebase=False)

with gr.Blocks() as demo:
    gr.Markdown("# Chat with your local Knowledge Base")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=640)
            message = gr.Textbox(placeholder="Ask me everything", label="Input")

            with gr.Row():
                clear = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=1):
            files = gr.File(label="Upload Knowledge Base",
                            file_types=['.txt', '.md', '.docx', '.pdf', '.pptx', '.epub', '.xlsx'])
            with gr.Row():
                upload = gr.Button("Upload")
                with_knowledgebase = gr.Radio(["Yes", "No"],
                                              label="Use Knowledge Base",
                                              value="No")
            with gr.Accordion(label="Models Configs", open=False):
                llm_name = gr.Dropdown(
                    choices=llm_list,
                    label="LLM",
                    value=default_llm,
                    interactive=True
                )
                embedding_name = gr.Dropdown(
                    choices=embedding_list,
                    label="Embedding Model",
                    value=DEFAULT_EMBEDDING,
                    interactive=True
                )
                temperature = gr.Slider(0, 1, value=0.1, step=0.1,
                                        label="Temperature", interactive=True)
                top_k = gr.Slider(1, 10, value=3, step=1,
                                  label="Vector db retrieve top k", interactive=True)

    # Event
    def respond(
            user_message,
            chat_history,
            conf_model,
            conf_embedding,
            conf_temperature,
            conf_top_k,
            conf_with_knowledgebase):
        _with_knowledgebase = True if conf_with_knowledgebase == "Yes" else False
        chat_history.append([user_message, None])
        yield "", chat_history
        bot_message = chain.send(user_message,
                                 conf_model, conf_embedding,
                                 conf_temperature,
                                 conf_top_k,
                                 _with_knowledgebase)
        chat_history[-1][1] = ""
        for character in bot_message:
            chat_history[-1][1] += character
            yield "", chat_history


    def clear_history(history):
        chain.clear_history()
        history.clear()
        return history


    def upload_kb(file_paths):
        gr.Info("Uploading KnowledgeBase...")
        chain.upload_kb(file_paths)
        gr.Info("Upload Successfully!")


    message.submit(
        respond,
        [message, chatbot, llm_name, embedding_name, temperature, top_k, with_knowledgebase],
        [message, chatbot]
    )
    clear.click(clear_history, chatbot, chatbot)
    upload.click(upload_kb, inputs=files, outputs=[])

if __name__ == "__main__":
    demo.launch()
