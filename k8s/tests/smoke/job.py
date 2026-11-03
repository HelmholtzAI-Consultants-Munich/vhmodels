"""Ray job submitted through the port-forwarded Jobs endpoint."""

import socket

import ray

ray.init()
print("cluster resources:", ray.cluster_resources())


@ray.remote
def square(x):
    return x * x, socket.gethostname()


results = ray.get([square.remote(i) for i in range(8)])
assert [r[0] for r in results] == [i * i for i in range(8)]
print("hosts:", sorted({r[1] for r in results}))
print("SMOKE JOB OK")
