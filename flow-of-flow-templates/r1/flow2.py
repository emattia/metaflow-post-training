from metaflow import (
    FlowSpec,
    step,
    current,
    Parameter,
    card,
    gpu_profile,
    model,
    environment,
    IncludeFile,
    huggingface_hub,
    checkpoint,
    kubernetes,
    parallel,
    secrets,
    trigger_on_finish
)

@trigger_on_finish(flow='SFTReasoning')
class GRPO(FlowSpec):

    training_config = IncludeFile(
        "config",
        default="grpo_config.yaml", 
        is_text=True,
    )
    recipe = Parameter(
        "recipe",
        default="full_grpo_distributed.py",
        help="The name of the recipe or .py file that defines the recipe. Metaflow will automatically package .py files in the flow directory."
    )

    @step
    def start(self):
        self.next(self.train, num_parallel=4)

    @parallel
    @step
    def train(self):
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    GRPO()