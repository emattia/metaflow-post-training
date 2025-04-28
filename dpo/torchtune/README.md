## Setup 

1. Install Python 3.12
2. `pip install -r dev-reqs.txt`

## Manual run

### Download a model using torchtune

[Read more](https://pytorch.org/torchtune/stable/tutorials/first_finetune_tutorial.html#downloading-a-model).
```bash
tune download meta-llama/Llama-3.3-70B-Instruct --output_dir $BASE_MODEL_DIR
```

### DPO fine-tuning

Check out the [`dpo_config.yaml`](./dpo_config.yaml) file and corresponding [`dpo_recipe.py`](./dpo_recipe.py).
These files work together to drive the torchtune fine-tuning process.
Here we run them independent of Metaflow to highlight the separation of concerns.

```bash
tune run --nproc_per_node 8 dpo_recipe.py --config dpo_config.yaml 
```

The config file has a parameter called `output_dir`. 
This determines where the tuned model weights are saved to disk. 

### DPO 

To then run inference, we choose the emerging standard vLLM framework. 
You could replace with any framework you like.

The only required parameter to this custom eval implementation is the path to the local file containing

```bash
DPO_MODEL_DIR=llama3_3_70B/full_dpo/epoch_0/ # TODO: change to your path
python dpo_eval.py --checkpoint-path $DPO_MODEL_DIR
```

## Run with Metaflow

Finally, we have prepared a workflow that will run both training and evaluation in separate Kubernetes steps,
demonstrating how to move model assets and datasets across task runtimes.

```bash
python flow.py --environment=fast-bakery run
```