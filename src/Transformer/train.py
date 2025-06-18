import torch
from torch import nn
from tqdm import tqdm

from src.Transformer.Dataset.WikipediaDataset import WikipediaDataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from src.Transformer.Model.GPTLikeLM import GPTLikeLM

# ---- HYPERPARAMS ----
SEQ_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 8
D_FF = 1024
DROPOUT = 0.1
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- TOKENIZER ----
tokenizer = Tokenizer.from_file("Dataset/polish-bpe-tokenizer.json")
VOCAB_SIZE = tokenizer.get_vocab_size()

SEQ_LEN = 128
with open("Dataset/wiki.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if len(line.strip()) > 0]
dataset = WikipediaDataset(texts, tokenizer, seq_len=SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ---- MODEL ----
model = GPTLikeLM(
    vocab_size=VOCAB_SIZE,
    seq_length=SEQ_LEN,
    d_model=D_MODEL,
    N=N_LAYERS,
    h=N_HEADS,
    dropout=DROPOUT,
    d_ff=D_FF,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"), reduction="mean")

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)

# ---- TRAINING LOOP ----
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoka {epoch+1}/{EPOCHS}"):
        x, y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        mask = generate_square_subsequent_mask(SEQ_LEN).to(DEVICE)
        logits = model(x, mask)              # (B, T, V)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoka {epoch+1}: Średnia strata: {avg_loss:.4f}")
    # Save model checkpoint
    torch.save(model.state_dict(), f"gpt_polish_epoch{epoch+1}.pt")

print("Trening zakończony!")