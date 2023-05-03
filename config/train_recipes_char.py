# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = "out-recipes-char-2"
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True  # override via command line if you like
wandb_project = "recipe-char"
wandb_run_name = "recipe-mini-gpt"

dataset = "recipes"
gradient_accumulation_steps = 1
batch_size = 32
block_size = 64  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 128
# dropout = 0.2

# --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
# python train.py config/train_recipes_char.py --block_size=32 --batch_size=128 --n_layer=16 --n_head=16 --n_embd=256 --max_iters=4000 --lr_decay_iters=4000 --dropout=0.01
# learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 4000
lr_decay_iters = 4000  # make equal to max_iters usually
# min_lr = 1e-4  # learning_rate / 10 usually
# beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
device = "mps"  # run on cpu only
compile = False  # do not torch compile the model
