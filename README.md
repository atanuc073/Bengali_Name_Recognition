
# Bengali Name Entity Recognition

This a project to extract names names in a bengali text.

## Authors

- [Atanu Chowdhury](www.linkedin.com/in/atanu-chowdhury)



## Data Used

- [Dataset 1](https://github.com/Rifat1493/Bengali-NER/tree/master/annotated%20data)
- [Dataset 2](https://raw.githubusercontent.com/banglakit/bengali-ner-data/master/main.jsonl)

## Model Building

I have developed 3 model to approach this problem.



## Model 1 - BERT Embedding -> MLP  NER Model

For the first model, the approach is to develop a simple NER model that takes the text data sequentially and passes it through a pretrained BERT embedding layer to get the embedding of the text.

### Model Architecture

The model architecture can be summarized as follows:

1. Input: Sequential text data (sentences or phrases).
2. Pretrained BERT Embedding Layer: The input text is passed through a pretrained BERT (Bidirectional Encoder Representations from Transformers) embedding layer, which captures contextualized word embeddings.
3. NER Layer: The BERT embeddings are then passed through a final MLP  layer and softmax layer, which maps the embeddings to named entity tags (e.g., PERSON, ORGANIZATION, LOCATION, etc.).

### Advantages

- BERT embeddings capture contextual information and semantic meaning of words, improving NER performance.
- Transfer learning from a pretrained BERT model reduces the need for extensive training data.

### Limitations

- Training and inference times can be relatively high due to the complexity of the BERT model.
- Requires a significant amount of computational resources, especially for large models.


### Code Example

```python
from transformers import AutoModelForPreTraining, AutoTokenizer
class EntityModel(nn.Module):
    def __init__(self,num_tag,bert_model=bert_model):
        super(EntityModel,self).__init__()
        self.num_tag=num_tag
        self.bert=transformers.BertModel.from_pretrained(bert_model)
#         for param in self.bert.parameters():
#             param.requires_grad=False
        self.drop1=nn.Dropout(0.3)
#         self.hidden1=nn.Linear(768,256)
        self.out_tag=nn.Linear(768,num_tag)
        
    
    def forward(self,ids,mask,token_type_ids,target_tag):
        o1=self.bert(ids,attention_mask=mask,token_type_ids=token_type_ids)
        o1=o1["last_hidden_state"]
        bo_tag=self.drop1(o1)

#         bo_tag=self.hidden1(bo_tag)
        tag=self.out_tag(bo_tag)


        loss=loss_fn(tag,target_tag,mask,self.num_tag)

        return tag,loss
```
## Model 2 BERT Embedding -> Bi directional LSTM -> MLP softmax Model

For this model, the approach is to leverege the power of Bi Directional LSTM to find the meaning ful context of the word in a sentence .

### Model Architecture

The model architecture can be summarized as follows:

1. Input: Sequential text data (sentences or phrases).
2. Pretrained BERT Embedding Layer: The input text is passed through a pretrained BERT (Bidirectional Encoder Representations from Transformers) embedding layer, which captures contextualized word embeddings.
3. The BERT embedding are passed through the Bi directional LSTM to capture the long texts.
3. NER Layer: The Output of the Bi directional LSTM are then passed through a final MLP  layer and softmax layer, which maps the embeddings to named entity tags (e.g., PERSON, ORGANIZATION, LOCATION, etc.).

### Advantages

- Adding to the First model Approach adding Bi LSTM will benefit in long text sentences , It will reduce the gradient descent problem.
### Limitations
- WE have very small dataset which is not enough to get the full potential of Bi directional LSTM.
- Training and inference times can be relatively high due to the complexity of the BERT and Bi LSTM model.
- Requires a significant amount of computational resources, especially for large models.
### Code Example

```python
class Bert_BiLSTM_MLP(nn.Module):
    def __init__(self, n_tags, hidden_dim=50,num_heads=1, bert_model=bert_model):
        super(Bert_BiLSTM_MLP, self).__init__()
        self.num_tag=n_tags
        self.bert = transformers.BertModel.from_pretrained(bert_model)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.out=nn.Linear(hidden_dim * 2,n_tags)


    def forward(self, ids,mask,token_type_ids,target_tag):
        bert_output = self.bert(ids,attention_mask=mask,token_type_ids=token_type_ids)
        bert_output=bert_output["last_hidden_state"]
        bi_lstm_out, _ = self.lstm(bert_output)
        out=self.out(bi_lstm_out)
        loss=loss_fn(out,target_tag,mask,self.num_tag)
        return out,loss
```
## Model 3 BERT Embedding -> Attention Mechanism-> MLP softmax Model

For this model, the approach is to leverege the power of State of the art Attention Mechanism to find the meaningful context of the word in a sentence .

### Model Architecture

The model architecture can be summarized as follows:

1. Input: Sequential text data (sentences or phrases).
2. Pretrained BERT Embedding Layer: The input text is passed through a pretrained BERT (Bidirectional Encoder Representations from Transformers) embedding layer, which captures contextualized word embeddings.
3. The BERT embedding are passed through Attention layer to find which other words the model should focus more when predicting the tag for target word.
3. NER Layer: The Output of the Bi directional LSTM are then passed through a final MLP  layer and softmax layer, which maps the embeddings to named entity tags (e.g., PERSON, ORGANIZATION, LOCATION, etc.).

### Advantages

- Adding to the First  and second model Approach addingAttention model will benefit in caputring the semantic value of the word with respect to other words in the text sentences.
### Limitations
- WE have very small dataset which is not enough to get the full potential of Attention model.
- Training and inference times can be relatively high due to the complexity of the BERT and Bi LSTM model.
- Requires a significant amount of computational resources, especially for large models.
### Code Example

```python
class BertAttentionMLP(nn.Module):
    def __init__(self, n_tags, hidden_dim=50,num_heads=1, bert_model=bert_model):
        super(BertAttentionMLP, self).__init__()
        self.num_tag=n_tags
        self.bert = transformers.BertModel.from_pretrained(bert_model)
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(self.bert.config.hidden_size, num_heads=num_heads)
#         self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.out=nn.Linear(hidden_dim,n_tags)
#         self.crf = CRF(n_tags)

    def forward(self, ids,mask,token_type_ids,target_tag):
        bert_output = self.bert(ids,attention_mask=mask,token_type_ids=token_type_ids)
        bert_output=bert_output["last_hidden_state"]
        attention_output, _ = self.attention(bert_output, bert_output, bert_output)
#         print(attention_output.shape)
        dense_output = self.fc(attention_output)
#         crf_output = self.crf(dense_output)
        out=self.out(dense_output)
        loss=loss_fn(out,target_tag,mask,self.num_tag)
        return out,loss
```

## Model Performance


| Model                | Train Loss | Train Accuracy | Train F1 Score | Valid Loss | Valid Accuracy | Valid F1 Score |
|----------------------|------------|----------------|----------------|------------|----------------|----------------|
| EntityModel          | 0.083      | 0.970          | 0.970          | 0.179      | 0.936          | 0.934          |
| BI LSTM MLP          | 0.061      | 0.983          | 0.983          | 0.189      | 0.946          | 0.946          |
| Bert Attention MLP   | 0.043      | 0.985          | 0.985          | 0.200      | 0.947          | 0.946          |

## Acknowledgements

 - [Pretrained BERT model](https://huggingface.co/sagorsarker/bangla-bert-base)



## Model Output
## Screenshots

![App Screenshot]

