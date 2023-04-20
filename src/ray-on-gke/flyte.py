import typing

import ray
from flytekit import task
from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig


@ray.remote
def f(x):
    return x * x


@task(
    task_config=RayJobConfig(
        head_node_config=HeadNodeConfig(ray_start_params={"block": "true"}),
        worker_node_config=[
            WorkerNodeConfig(
                group_name="ray-group", replicas=5, min_replicas=2, max_replicas=10
            )
        ],
        runtime_env={"pip": ["numpy", "pandas"]},
    )
)
def ray_task(n: int) -> typing.List[int]:
    futures = [f.remote(i) for i in range(n)]
    return ray.get(futures)
