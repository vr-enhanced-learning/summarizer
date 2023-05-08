import gradio as gr
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

# Load the model and tokenizer from Hugging Face Spaces
model_name = "currentlyexhausted/flan-t5-summarizer"  # update with your model name
config = T5Config.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)

# Define the function to generate summary
def generate_summary(passage):
    # Set the input text and convert it to token ids
    input_text = f'Generate summary of: "{passage}"'
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate the output with modified settings
    output = model.generate(
        input_ids=input_ids,
        max_length=512,  # Increase max_length
        num_beams=8,  # Increase num_beams
        no_repeat_ngram_size=4,  # Increase no_repeat_ngram_size
        early_stopping=True,
        top_p=0.95, # Sample from the top 95% of the distribution
        temperature=0.5 # Control the level of randomness
    )

    # Convert the output token ids to text
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text

# Create the Gradio interface
inputs = gr.inputs.Textbox(label="Enter the passage to summarize")
outputs = gr.outputs.Textbox(label="Summary")
title = "T5 Summarizer"
description = "Enter a passage and the T5 model will generate a summary based on the input passage."
examples = [["""While cockroaches are often considered to be the winners in a post-nuclear world, fungi may actually be the ones thriving in highly radioactive conditions. Melanin-rich fungi found inside the Chernobyl reactor are attracted to radiation and use their melanin as a shield while also absorbing energy from the radiation to enhance their growth. These fungi could potentially be used to clean up radioactive waste and develop new cancer treatments. Therefore, if there is a nuclear accident, fungi might be able to survive and help clean up the aftermath."""], [""""How long does it take for the Earth to go around the sun? Well, a day. \"Well\" Yeah? \"Twentyfour hours.\" Twenty four hours? What do you reckon, cuz? Isn't it twenty four hours? Obviously a day, yes. A day, I hope. I know it takes, uh, ninety minutes for the space station to go around the earth Threehundred and sixty days. No, one day. Is it one day? I don't know. What is it? I say one year. I dunno, like a year. Uh, a year. One year. Twelve months. Twelve months? Threehundred and sixty five Threehundred and sixtyfive Threehundred and sixtyfive Threehundred and sixtyfive Threehundred and sixty five days? Yeah, I'd agree with that Threesixty five. It's somewhere between that, it's not quite precise, is it? Threehundred and sixtyfive and a bit days. It's nearly three hundred andit's three hundred and sixty five Five and a quarter days. Wait, the earth doesn't take one day to get around the sun. Takes about a year.]\""""]]
gr.Interface(fn=generate_summary, inputs=inputs, outputs=outputs, title=title, description=description, examples=examples).launch()
