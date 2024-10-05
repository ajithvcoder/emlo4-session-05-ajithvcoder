### EMLOV4-Session-05 Assignment - PyTorch Lightning - II

### PyTorch Lightning - II

#### Requirements:

- Start from the repository of previous Session
- Start using Cursor!
- Create a eval.py  with its config that tests the model given a checkpoint
- Integrate the infer.py  you made in last session with hydra
- Make sure to integrate codecov in your repo, should be atleast 70%
- Push the Docker Image to GHCR, should show up in Packages section of your repo


### Development method

- Write configs with hydra - do today
- write test and use testcoverage - do today
- contanarise and write ci to push to github registry

- python src/eval.py callbacks.model_checkpoint.filename='model_storage/epoch0-checkpoint.ckpt'
- python src/infer.py callbacks.model_checkpoint.filename='model_storage/epoch-checkpoint.ckpt'
### Build Command

test coverage 
pytest --cov-report term --cov=src/models/ tests/models/test_timm_classifier.py
pytest --cov-report term --cov=src/ tests/


```
docker build -t dog_train -f ./Dockerfile .
```
docker run -d -v /workspace/emlo4-session-05-ajithvcoder/:/workspace/ dog_train   tail -f /dev/null
^C^C^C
docker exec -it 2f2ee3466b56 /bin/bash


python src/infer.py  --input_folder data/dataset/val/ --output_folder infer_images --ckpt_path "model_storage/epoch-checkpoint.ckpt"

### Docker file usage to train, eval and infer
- Train

```
docker run --rm -v ./model_storage:/workspace/model_storage dog_train python src/train.py --data data --logs logs --ckpt_path model_storage 
```

- Eval

```
docker run --rm -v ./model_storage:/workspace/model_storage dog_train python src/eval.py --data data --ckpt_path "model_storage/epoch=0-checkpoint.ckpt"
```

- Infer

```
docker run --rm -v ./model_storage:/workspace/model_storage -v ./infer_images:/workspace/infer_images dog_train python src/infer.py  --input_folder data/dataset/val/ --output_folder infer_images --ckpt_path "model_storage/epoch-checkpoint.ckpt"
```

### Learnings:

### Group Members
1. Ajith Kumar V (myself)
2. Aakash Vardhan
3. Anvesh Vankayala
4. Manjunath Yelipeta
5. Abhijith Kumar K P
6. Sabitha Devarajulu