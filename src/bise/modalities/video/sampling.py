"""Put both cameras of two distinct tasks in each four-pair training batch."""

import heapq
import random
from collections import defaultdict

from torch.utils.data import Sampler, Subset


def subset_samples(dataset):
    """Read sample metadata in local index order, without decoding videos."""
    if isinstance(dataset, Subset):
        samples = subset_samples(dataset.dataset)
        return [samples[index] for index in dataset.indices]
    return dataset.samples


class SceneViewBatchSampler(Sampler):
    def __init__(self, dataset, seed=44):
        self.seed = int(seed)
        self.epoch = 0
        groups = defaultdict(list)
        samples = subset_samples(dataset)
        for index, sample in enumerate(samples):
            groups[(sample.task_id, sample.scene_id)].append(index)
        self.tasks = defaultdict(list)
        for (task, scene), indices in sorted(groups.items()):
            cameras = [samples[index].camera_id for index in indices]
            if len(indices) != 2 or len(set(cameras)) != 2:
                raise ValueError(f"scene_views requires exactly two cameras per scene: {scene}")
            self.tasks[task].append(sorted(indices, key=lambda i: samples[i].camera_id))
        self.scene_count = len(groups)
        if self.scene_count < 2 or self.scene_count % 2:
            raise ValueError("scene_views requires a positive even number of training scenes")
        if max(map(len, self.tasks.values())) > self.scene_count // 2:
            raise ValueError("scene_views cannot pair every scene with a different task")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.scene_count // 2

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        tasks = {task: list(scenes) for task, scenes in self.tasks.items()}
        for scenes in tasks.values():
            rng.shuffle(scenes)
        # Largest remaining groups first avoid leaving an unpairable task at the end.
        heap = [(-len(scenes), rng.random(), task) for task, scenes in tasks.items()]
        heapq.heapify(heap)
        batches = []
        while heap:
            left, right = heapq.heappop(heap), heapq.heappop(heap)
            batch = tasks[left[2]].pop() + tasks[right[2]].pop()
            rng.shuffle(batch)
            batches.append(batch)
            for count, _, task in (left, right):
                if count < -1:
                    heapq.heappush(heap, (count + 1, rng.random(), task))
        rng.shuffle(batches)
        yield from batches
