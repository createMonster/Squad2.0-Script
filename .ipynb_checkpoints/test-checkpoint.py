from transformers import AutoTokenizer, AutoModel  
tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")  
model = AutoModel.from_pretrained("SpanBERT/spanbert-large-cased")

print (model)