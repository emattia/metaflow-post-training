from metaflow import FlowSpec, step, kubernetes, pypi

class GPUTestFlow(FlowSpec):

    @kubernetes(gpu=1, image="docker.io/eddieob/bert-gpu-example")
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
    GPUTestFlow()