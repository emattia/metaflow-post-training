import os
import json
import time
import torch
import logging
from functools import partial
from typing import Dict, Optional, Any
from tqdm import tqdm

from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, padded_collate_dpo
from torchtune import config, rlhf
from torchtune.rlhf.loss import DPOLoss
from torchtune.config._utils import _get_component_from_path
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_tokenizer_path(checkpoint_path: str, custom_tokenizer_path: Optional[str] = None) -> str:
    """
    Get the tokenizer directory path based on TorchTune's standard structure.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        custom_tokenizer_path: Optional custom path to the tokenizer
        
    Returns:
        Path to the tokenizer directory
    """
    if custom_tokenizer_path:
        if os.path.isfile(custom_tokenizer_path):
            return os.path.dirname(custom_tokenizer_path)
        return custom_tokenizer_path
    
    # Standard TorchTune structure: /path/to/model/original/tokenizer.model
    # Return the directory containing the tokenizer
    return os.path.join(checkpoint_path, "original")


def run_eval(
    checkpoint_path: str,

    # 🚨 The next two params should be match any custom dataset.
    data_component: str = 'torchtune.datasets.stack_exchange_paired_dataset', 
    data_split: str = 'train',

    # For testing, below this can stay constant. Tune hyperparameters for real batch inference use cases.
    collate_fn = padded_collate_dpo,
    tokenizer_component: str = 'torchtune.models.llama3.llama3_tokenizer',
    tokenizer_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    world_size: int = 8,
    rank: int = 0,
    batch_size: int = 2,
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    max_batches: Optional[int] = 10,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    
    
    """
    Run evaluation on a model checkpoint using vLLM for inference.
    
    This function evaluates a model on DPO-style paired datasets, calculating preference scores
    and comparing policy against reference models.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        data_component: The dataset component path to use
        data_split: Partition of the dataset
        collate_fn: Function to collate batch items
        tokenizer_component: The tokenizer component path
        tokenizer_path: Optional custom path to the tokenizer
                      (if not provided, uses checkpoint_path/original)
        output_dir: Directory to save detailed results (None for no saving)
        world_size: Number of GPUs for tensor parallelism
        rank: Process rank for distributed evaluation
        batch_size: Batch size for evaluation
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_batches: Maximum number of batches to evaluate (None for all)
        seed: Random seed
        verbose: Whether to display progress bar
        
    Returns:
        Dictionary containing evaluation results
    """

    start_time = time.time()
    is_main_process = rank == 0
    
    torch.manual_seed(seed)

    if output_dir and is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Results will be saved to {output_dir}")

    if tokenizer_path:
        tokenizer_dir = get_tokenizer_path(checkpoint_path, tokenizer_path)
    else: 
        tokenizer_dir = get_tokenizer_path(checkpoint_path)
    tokenizer_model_path = os.path.join(tokenizer_dir, "tokenizer.model")
    
    if not os.path.exists(tokenizer_model_path):
        logger.warning(f"Expected tokenizer not found at {tokenizer_model_path}")
        if os.path.exists(tokenizer_dir):
            logger.info(f"Files in tokenizer directory: {os.listdir(tokenizer_dir)}")
    
    logger.info(f"Using tokenizer directory: {tokenizer_dir}")
    
    ### Load assets
    cfg_tokenizer = DictConfig({
        '_component_': tokenizer_component,
        'path': tokenizer_model_path,
        'max_seq_len': None
    })
    logger.info("Loading tokenizer for dataset processing...")
    tokenizer = config.instantiate(cfg_tokenizer)
    logger.info("Loading dataset...")
    cfg_dataset = DictConfig({
        '_component_': data_component,
        'split': data_split
    })
    dataset = config.instantiate(cfg_dataset, tokenizer)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed
        )
    logger.info("Setting up dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=partial(
            collate_fn,
            padding_idx=tokenizer.pad_id,
            ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
        )
    )
    logger.info(f"Loading policy model from {checkpoint_path}...")
    try:
        policy_model = LLM(
            model=checkpoint_path,
            trust_remote_code=True,
            tensor_parallel_size=world_size,
            dtype="bfloat16",
            gpu_memory_utilization=0.9
        )
    except Exception as e:
        logger.error(f"Failed to load policy model: {str(e)}")
        raise
    
    # Setup stop tokens
    # if hasattr(tokenizer, "stop_tokens"):
    #     stop_token_ids = tokenizer.stop_tokens
    # else:
    #     stop_token_ids = [tokenizer.eos_id]
    
    # Setup sampling parameters
    sampling_kwargs = dict(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        # logprobs=1
    )
    sampling_params = SamplingParams(**sampling_kwargs)
    logger.info(f"Starting evaluation with sampling params: {sampling_kwargs}...")
    
    all_results = {
        "metadata": {
            "checkpoint_path": checkpoint_path,
            "tokenizer_path": tokenizer_model_path,
            "data_split": data_split,
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "world_size": world_size,
            "rank": rank,
            "seed": seed,
        },
        "samples": []
    }
    
    num_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    iter_dataloader = enumerate(dataloader)
    logger.info(f'Evaluating on {num_batches} batches')
    
    if verbose and is_main_process:
        iter_dataloader = tqdm(iter_dataloader, total=num_batches, desc="Evaluating batches")

    for batch_idx, batch in iter_dataloader:
        if max_batches is not None and batch_idx >= max_batches:
            break
        concatenated_input_ids, concatenated_labels = batch
        concatenated_input_ids = concatenated_input_ids.to('cuda')
        concatenated_labels = concatenated_labels.to('cuda')
        prompt_token_ids = concatenated_input_ids.tolist()
        vllm_inp = [TokensPrompt(prompt_token_ids=seq) for seq in prompt_token_ids]
        outputs = policy_model.generate(vllm_inp, sampling_params, use_tqdm=False)
        for i, output in enumerate(outputs):
            sample = {}
            sample["prompt_text_inputs"] = tokenizer.decode(prompt_token_ids[i])
            sample["gen_text_outputs"] = output.outputs[0].text
            all_results['samples'].append(sample)

    execution_time = time.time() - start_time
    all_results["execution_time"] = execution_time
    
    if output_dir and is_main_process:
        output_file = os.path.join(output_dir, f"dpo_eval_results_rank{rank}.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
 
    if is_main_process:
        logger.info(f"Evaluation completed in {all_results['execution_time']:.2f} seconds")

    return all_results

def parse_arguments():
    """Parse command line arguments for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation on a model checkpoint")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the model checkpoint")
    # parser.add_argument("--data-split", help="Path to the dataset")
    parser.add_argument("--tokenizer-path", help="Optional override path to the tokenizer directory")
    parser.add_argument("--output-dir", help="Directory to save detailed results")
    parser.add_argument("--world-size", type=int, default=8, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--rank", type=int, default=0, help="Process rank for distributed evaluation")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--max-batches", type=int, default=10, help="Maximum number of batches to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Display progress bar")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    results = run_eval(
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        world_size=args.world_size,
        rank=args.rank,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_batches=args.max_batches,
        seed=args.seed,
        verbose=args.verbose
    )
    
    if args.rank == 0 and args.output_dir is None:
        logger.info(f"Evaluation summary: {json.dumps(results['samples'][:3], indent=2)}")