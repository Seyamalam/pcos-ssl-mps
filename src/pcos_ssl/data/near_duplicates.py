from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


@dataclass
class BKNode:
    value: str
    indices: list[int] = field(default_factory=list)
    children: dict[int, "BKNode"] = field(default_factory=dict)


class BKTree:
    def __init__(self) -> None:
        self.root: BKNode | None = None

    def add(self, value: str, index: int) -> None:
        if self.root is None:
            self.root = BKNode(value=value, indices=[index])
            return

        node = self.root
        while True:
            distance = hamming_hex(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value=value, indices=[index])
                return
            node = child

    def query(self, value: str, threshold: int) -> list[int]:
        if self.root is None:
            return []
        matches: list[int] = []
        nodes = [self.root]
        while nodes:
            node = nodes.pop()
            distance = hamming_hex(value, node.value)
            if distance <= threshold:
                matches.extend(node.indices)
            lower = distance - threshold
            upper = distance + threshold
            nodes.extend(child for edge, child in node.children.items() if lower <= edge <= upper)
        return matches


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def build_phash_groups(images: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    readable = images[images["readable"] & images["phash"].notna()].reset_index(drop=True)
    tree = BKTree()
    uf = UnionFind(len(readable))

    for idx, phash in enumerate(readable["phash"]):
        for match_idx in tree.query(phash, threshold=threshold):
            uf.union(idx, match_idx)
        tree.add(phash, idx)

    group_roots = [uf.find(index) for index in range(len(readable))]
    root_to_id = {root: f"phash_{group_id:06d}" for group_id, root in enumerate(sorted(set(group_roots)))}
    grouped = readable[["image_id", "rel_path", "class_name", "label", "exact_group_id", "phash"]].copy()
    grouped["near_group_id"] = [root_to_id[root] for root in group_roots]

    summary = (
        grouped.groupby("near_group_id")
        .agg(
            group_size=("image_id", "size"),
            classes=("class_name", lambda values: ",".join(sorted(set(values)))),
            labels=("label", "nunique"),
            representative_rel_path=("rel_path", "first"),
        )
        .reset_index()
        .sort_values(["group_size", "near_group_id"], ascending=[False, True])
    )
    return grouped.merge(summary, on="near_group_id", how="left")

