from metaflow import FlowSpec, step, kubernetes, pypi

class CoreweaveCPUTestFlow(FlowSpec):

    @kubernetes(compute_pool='')
    @step
    def start(self):
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    CoreweaveCPUTestFlow()