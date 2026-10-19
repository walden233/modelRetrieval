"""Organize video pairs by execution views or task identity."""

import heapq
import math
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


class TaskPairBatchSampler(Sampler):
    """Two distinct samples per task; cover all train samples and pad the last batches."""

    def __init__(self, dataset, batch_size=22, seed=44):
        if batch_size < 4 or batch_size % 2:
            raise ValueError("task_pairs requires an even batch_size of at least 4")
        self.samples = subset_samples(dataset)
        self.seed, self.epoch = int(seed), 0
        self.tasks_per_batch = batch_size // 2
        self.tasks = defaultdict(list)
        for index, sample in enumerate(self.samples):
            self.tasks[sample.task_id].append(index)
        if len(self.tasks) < self.tasks_per_batch:
            raise ValueError("task_pairs needs at least batch_size/2 distinct training tasks")
        if any(len(indices) < 2 for indices in self.tasks.values()):
            raise ValueError("task_pairs needs two distinct samples for every training task")
        pair_counts = [math.ceil(len(indices) / 2) for indices in self.tasks.values()]
        self.batch_count = max(math.ceil(sum(pair_counts) / self.tasks_per_batch),
                               max(pair_counts))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batch_count

    def _pairs(self, indices, rng):
        pools = defaultdict(list)
        for index in indices:
            pools[self.samples[index].scene_id].append(index)
        for pool in pools.values():
            rng.shuffle(pool)
        pairs = []
        while pools:
            # Prefer different scenes; use another camera when only one scene remains.
            scenes = sorted(pools, key=lambda scene: (-len(pools[scene]), rng.random()))
            first = pools[scenes[0]].pop()
            if not pools[scenes[0]]:
                del pools[scenes[0]]
            if len(scenes) > 1:
                other = scenes[1]
                second = pools[other].pop()
                if not pools[other]:
                    del pools[other]
            elif pools:
                second = pools[scenes[0]].pop()
                if not pools[scenes[0]]:
                    del pools[scenes[0]]
            else:
                second = rng.choice([index for index in indices if index != first])
            pairs.append([first, second])
        return pairs

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pairs = {task: self._pairs(indices, rng) for task, indices in sorted(self.tasks.items())}
        original = {task: list(values) for task, values in pairs.items()}
        heap = [(-len(values), rng.random(), task) for task, values in pairs.items()]
        heapq.heapify(heap)
        batches = []
        for _ in range(len(self)):
            selected = [heapq.heappop(heap) for _ in range(min(self.tasks_per_batch, len(heap)))]
            used = {task for _, _, task in selected}
            batch = [index for _, _, task in selected for index in pairs[task].pop()]
            padding = sorted(set(self.tasks) - used)
            for task in rng.sample(padding, self.tasks_per_batch - len(selected)):
                batch.extend(rng.choice(original[task]))
            for count, _, task in selected:
                if count < -1:
                    heapq.heappush(heap, (count + 1, rng.random(), task))
            rng.shuffle(batch)
            batches.append(batch)
        rng.shuffle(batches)
        yield from batches


class ScenePairBatchSampler(TaskPairBatchSampler):
    """One scene with both cameras per task; keep batches full and cover all scenes."""

    def __init__(self, dataset, batch_size=22, seed=44):
        super().__init__(dataset, batch_size, seed)
        for indices in self.tasks.values():
            self._pairs(indices, random.Random(seed))

    def _pairs(self, indices, rng):
        scenes = defaultdict(list)
        for index in indices:
            scenes[self.samples[index].scene_id].append(index)
        pairs = []
        for scene, group in sorted(scenes.items()):
            cameras = {self.samples[index].camera_id for index in group}
            if len(group) != 2 or len(cameras) != 2:
                raise ValueError(f"scene_pairs requires exactly two cameras per scene: {scene}")
            pairs.append(sorted(group, key=lambda index: self.samples[index].camera_id))
        rng.shuffle(pairs)
        return pairs


def build_training_sampler(dataset, training, seed):
    """Return None for the existing random sampling path."""
    name = training.get("sampler", "random")
    if name == "random":
        return None
    if name == "scene_views":
        if training["batch_size"] != 4:
            raise ValueError("scene_views requires training.batch_size=4")
        return SceneViewBatchSampler(dataset, seed)
    if name == "task_pairs":
        return TaskPairBatchSampler(dataset, training["batch_size"], seed)
    if name == "scene_pairs":
        return ScenePairBatchSampler(dataset, training["batch_size"], seed)
    raise ValueError(f"Unknown video training sampler: {name}")
