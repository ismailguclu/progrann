FROM doduo1.umcn.nl/uokbaseimage/tensorflow_pytorch_python3:3
RUN pip install -r requirements.txt
CMD ["bash", "-c"]
