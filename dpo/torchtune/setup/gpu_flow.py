from metaflow import FlowSpec, step, kubernetes
from metaflow.profilers import gpu_profile

class CoreweaveGPUTestFlow(FlowSpec):

    @gpu_profile(interval=1)
    @kubernetes(gpu=1, image="docker.io/eddieob/bert-gpu-example", compute_pool='coreweave-h100')
    @step
    def start(self):
        import torch # pylint: disable=import-error

        if torch.cuda.is_available():
            print('Happy days!')
        else:
            print('Oh no, PyTorch cannot see CUDA on this machine.')
            exit()
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    CoreweaveGPUTestFlow()