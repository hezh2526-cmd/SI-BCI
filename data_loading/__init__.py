"""BCI2020 模型输入与 DataLoader 工具。"""

__all__ = [
    "BCIEpochDataset",
    "DataLoaderBundle",
    "create_dataloaders",
    "FewShotDataLoaderBundle",
    "create_few_shot_dataloaders",
]


def __getattr__(name):
    if name in {"FewShotDataLoaderBundle", "create_few_shot_dataloaders"}:
        from .bci2020_cbramod_few_shot import (
            FewShotDataLoaderBundle,
            create_few_shot_dataloaders,
        )
        return {
            "FewShotDataLoaderBundle": FewShotDataLoaderBundle,
            "create_few_shot_dataloaders": create_few_shot_dataloaders,
        }[name]
    if name in {"BCIEpochDataset", "DataLoaderBundle", "create_dataloaders"}:
        from .bci2020_data import BCIEpochDataset, DataLoaderBundle, create_dataloaders
        return {
            "BCIEpochDataset": BCIEpochDataset,
            "DataLoaderBundle": DataLoaderBundle,
            "create_dataloaders": create_dataloaders,
        }[name]
    raise AttributeError(name)
