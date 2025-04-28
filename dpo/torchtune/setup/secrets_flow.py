from metaflow import FlowSpec, step, secrets, pypi

class SecretsTestFlow(FlowSpec):

    
    # TODO: Fill in your secrets, after going to /Integrations tab on Outerbounds UI.
        # 1. [Optional] Create WANDB_API_KEY secret
        # 2. Create HUGGINGFACE_API_KEY secret
    @secrets(sources=["..."]) # TODO: Add your secret source here.
    # @pypi(packages={"wandb": "", "requests": ""})
    @step
    def start(self):

        # NOTE: uncomment next to lines for wandb
        # import wandb
        # wandb.login(verify=True)
        
        import requests
        import os
        
        hf_token = os.environ.get('HF_TOKEN', None)
        if hf_token is None:
            print('HUGGINGFACE_API_KEY is not set.')
        else:
            response = requests.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {hf_token}"}
            )
            print("Huggingface request seems to be working! Identified:")
            print(response.json())
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    SecretsTestFlow()