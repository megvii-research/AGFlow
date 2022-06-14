# C -> S
python evaluate.py --model=weights/agflow-chairs.pth  --dataset=sintel --mixed_precision
# C+T -> S
python evaluate.py --model=weights/agflow-things.pth  --dataset=sintel --mixed_precision