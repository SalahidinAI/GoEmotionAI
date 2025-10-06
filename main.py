import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from fastapi import FastAPI, Request
from pydantic import BaseModel
from torchtext.data import get_tokenizer
import uvicorn


class GoEmotionsModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, 28)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.fc(hidden[-1])
        return x


# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем словарь и токенизатор
vocab = torch.load("vocab.pth", weights_only=False)
tokenizer = get_tokenizer("basic_english")


def text_pipeline(text: str):
    return [vocab[i] for i in tokenizer(text)]


# Модель
model = GoEmotionsModel(len(vocab)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# FastAPI
text_app = FastAPI(title="Text")


class TextIn(BaseModel):
    text: str


@text_app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data["text"]

    tensor = torch.tensor(text_pipeline(text), dtype=torch.int64).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)

        # Multi-label классификация
        probs = torch.sigmoid(pred)
        preds = (probs > 0.3).int().cpu().numpy().tolist()[0]

    return {"Result": preds}


if __name__ == "__main__":
    uvicorn.run(text_app, host="127.0.0.1", port=8000)
