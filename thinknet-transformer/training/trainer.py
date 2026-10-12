import os
import math
import torch

from torch.utils.data import DataLoader

from model.transformer import Transformer
from configs.model_config import ModelConfig
from data import TextDataset
from model.transformer import Transformer


# ==============================
# CONFIG
# ==============================

TOKENS_FILE = "data/tokens.pt"

CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 4
SEQ_LEN = 1024

GRAD_ACCUM_STEPS = 8

EPOCHS = 3

LR = 3e-4

WEIGHT_DECAY = 0.1

SAVE_STEPS = 500


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Allow overrides from environment for one-click runs
def _env_override_int(name, default):
    try:
        v = os.environ.get(name)
        return int(v) if v is not None else default
    except Exception:
        return default

# apply overrides
BATCH_SIZE = _env_override_int('BATCH_SIZE', BATCH_SIZE)
SEQ_LEN = _env_override_int('SEQ_LEN', SEQ_LEN)
GRAD_ACCUM_STEPS = _env_override_int('GRAD_ACCUM_STEPS', GRAD_ACCUM_STEPS)
EPOCHS = _env_override_int('EPOCHS', EPOCHS)
SAVE_STEPS = _env_override_int('SAVE_STEPS', SAVE_STEPS)


# ==============================
# LR Scheduler
# ==============================


def cosine_schedule(
    optimizer,
    warmup_steps,
    total_steps
):

    def lr_lambda(step):

        if step < warmup_steps:
            return step / warmup_steps

        progress = (
            step - warmup_steps
        ) / (
            total_steps - warmup_steps
        )

        return 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


# ==============================
# Train
# ==============================


def train():

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )


    print("Loading tokens...")

    tokens = torch.load(
        TOKENS_FILE
    )


    dataset = TextDataset(
        tokens,
        SEQ_LEN,
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )


    config = ModelConfig()


    model = Transformer(
        config
    )

    model.to(DEVICE)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9,0.95),
    )


    total_steps = (
        len(loader)
        *
        EPOCHS
        //
        GRAD_ACCUM_STEPS
    )


    scheduler = cosine_schedule(
        optimizer,
        warmup_steps=500,
        total_steps=total_steps,
    )


    scaler = torch.cuda.amp.GradScaler()


    model.train()

    step = 0

    # Inference scheduling: run generation every 5% of total steps
    inference_interval = max(1, total_steps // 20) if total_steps > 0 else 1
    last_inference_step = 0


    def generate_from_tokens(start_tokens, gen_len=100):
        model.eval()

        with torch.no_grad():
            # ensure start_tokens is a 1D tensor on device
            if not isinstance(start_tokens, torch.Tensor):
                start = torch.tensor(start_tokens, dtype=torch.long)
            else:
                start = start_tokens.long()

            generated = start.unsqueeze(0).to(DEVICE)

            for _ in range(gen_len):
                context = (
                    generated[:, -config.max_seq_len :]
                    if generated.shape[1] > config.max_seq_len
                    else generated
                )

                logits = model(context)

                next_logits = logits[:, -1, :]
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

            return generated.squeeze(0).cpu().tolist()


    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch+1}/{EPOCHS}"
        )


        optimizer.zero_grad()


        for batch_idx, batch in enumerate(loader):

            x,y = batch


            x = x.to(
                DEVICE,
                non_blocking=True
            )

            y = y.to(
                DEVICE,
                non_blocking=True
            )


            with torch.cuda.amp.autocast(
                dtype=torch.float16
            ):

                logits = model(x)

                loss = torch.nn.functional.cross_entropy(
                    logits.view(
                        -1,
                        config.vocab_size
                    ),
                    y.view(-1)
                )


                loss = (
                    loss
                    /
                    GRAD_ACCUM_STEPS
                )


            scaler.scale(loss).backward()


            if (
                (batch_idx+1)
                %
                GRAD_ACCUM_STEPS
                ==
                0
            ):

                scaler.unscale_(
                    optimizer
                )


                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )


                scaler.step(
                    optimizer
                )

                scaler.update()


                scheduler.step()


                optimizer.zero_grad()


                step += 1


                # Periodic logging
                if step % 10 == 0:

                    print(
                        f"step {step} | loss {loss.item():.4f}"
                    )


                # Checkpointing
                if step % SAVE_STEPS == 0:

                    save_path = (
                        f"{CHECKPOINT_DIR}/"
                        f"model_{step}.pt"
                    )


                    torch.save(
                        {
                            "model":
                            model.state_dict(),

                            "optimizer":
                            optimizer.state_dict(),

                            "step":
                            step,
                        },
                        save_path,
                    )

                    print(
                        "Saved:",
                        save_path
                    )

                # Run inference every 5% progress
                if step - last_inference_step >= inference_interval:
                    try:
                        # use the first tokens as prompt (or random slice)
                        prompt_len = min(SEQ_LEN, tokens.numel()-1)
                        if prompt_len <= 0:
                            prompt = torch.randint(0, config.vocab_size, (1,))
                        else:
                            prompt = tokens[:prompt_len]

                        generated = generate_from_tokens(prompt, gen_len=100)

                        print(f"[Inference @ step {step}] Generated {len(generated)} tokens (ids):")
                        print(generated[-100:])

                    except Exception as e:
                        print("Inference failed:", e)

                    last_inference_step = step


if __name__ == "__main__":

    train()