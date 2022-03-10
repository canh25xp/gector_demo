import gradio

def hello(inp):
  return f"Hello {inp}!"

# For information on Interfaces, head to https://gradio.app/docs/
# For user guides, head to https://gradio.app/guides/
iface = gradio.Interface(
  fn=hello,
  inputs='text',
  outputs='text',
  title='Hello World', 
  description='The simplest Hosted interface.')  

iface.launch()