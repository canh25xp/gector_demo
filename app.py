import gradio
import os


def hello(inp):
    # Get the current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # List all files in the directory
    files = os.listdir(current_dir)
    # Print the list of files
    print("Files in the directory:")
    ret = ""
    for file in files:
        ret += file
        ret += "\n"
        print(file)

    return ret

# For information on Interfaces, head to https://gradio.app/docs/
# For user guides, head to https://gradio.app/guides/
# For Spaces usage, head to https://huggingface.co/docs/hub/spaces
iface = gradio.Interface(
    fn=hello,
    inputs="text",
    outputs="text",
    title="Hello World",
    description="The simplest interface!",
    allow_flagging="never"
)

iface.launch()
