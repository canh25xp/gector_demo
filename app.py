import sys
sys.path.append("gector")

from gector.gec_model import GecBERTModel
from huggingface_hub import hf_hub_download
import gradio as gr


def load(model_path = "models/roberta_1_gectorv2.th"):
    transformer_model = "roberta"
    special_tokens_fix = 1
    min_error_prob = 0.50
    confidence_bias = 0.20

    return GecBERTModel(
        vocab_path="gector/test_fixtures/roberta_model/vocabulary",
        model_paths=[model_path],
        max_len=50,
        min_len=3,
        iterations=5,
        min_error_probability=min_error_prob,
        lowercase_tokens=False,
        model_name=transformer_model,
        special_tokens_fix=special_tokens_fix,
        log=False,
        confidence=confidence_bias,
    )


def predict(lines, model, batch_size=32):
    test_data = [s.strip() for s in lines]  # Remove trailling spaces
    predictions = []
    batch = []
    cnt_corrections = 0
    for sent in test_data:
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    # output = '<eos>'.join([' '.join(x) for x in predictions])
    output = [" ".join(x) for x in predictions]
    return "\n".join(output)


# Gradio interface
title = "Gector web interface"
description = "Enter a text and select a model to correct grammar errors."

text_input = gr.Textbox(lines=5, label="Input text")

check_box = gr.Checkbox(label="Highlight output")

model_select = gr.Dropdown(
    ["GECToR-Roberta", "GECToR-XLNet", "T5-Large"], label="Select model"
)

output_text = gr.Textbox(lines=5, label="Output text")

examples = [
    [
        "He do this work well, but she ain't agree with him on that matter.",
        "GECToR-Roberta",
    ],
    [
        "Their going to the park to play baseball, and then we will be going out for dinner.",
        "GECToR-Roberta",
    ],
]


if __name__ == "__main__":
    model_path = hf_hub_download("canh25xp/GECToR-Roberta", "roberta_1_gectorv2.th")
    model = load(model_path)

    def get_prediction(text, model_name):
        if model_name != "GECToR-Roberta":
            return "Unsupported"

        output = predict([text], model)
        return output

    app = gr.Interface(
        fn=get_prediction,
        inputs=[text_input, model_select],
        outputs=output_text,
        title=title,
        description=description,
        examples=examples,
        allow_flagging="never"
    )

    app.launch(share=False)
