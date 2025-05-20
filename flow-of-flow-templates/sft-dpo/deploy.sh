#!/bin/bash

for flow in *flow*.py;
do
    python $flow argo-workflows create
done